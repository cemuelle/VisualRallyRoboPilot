import os
import pickle
import lzma
import glob
import numpy as np
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from data_collector_evaluate_pilot import DataCollectionEvaluatePilot
import torch
import torch.nn as nn
from ga_pipeline import genAl, get_pods
from kubernetes import client, config
from PyQt6 import QtCore, QtWidgets, QtGui


config.load_kube_config()

#variable
onlyOnce = True

# parameters
N_ITERATIONS = 5
GATE_WIDTH = 5
PORT = 7654

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(self.sigmoid(self.fc1(x)))
        x = self.dropout(self.sigmoid(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


class ExampleNNMsgProcessor():
    def __init__(self, controls):
        self.controls = controls
        self.always_forward = True
        self.model = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.load_state_dict(torch.load("models/MLP_model.pth", weights_only=True))
        #self.model.to(self.device)
        #self.model.eval()

    def nn_infer(self, message):
        forward,backward,left,right = False,False,False,False

        control_actu = self.controls.pop(0)

        forward = float(control_actu[0]) > 0.5
        backward = float(control_actu[1]) > 0.5
        left = float(control_actu[2]) > 0.5
        right = float(control_actu[3]) > 0.5

        return [
            ("forward", forward),
            ("back", backward),
            ("left", left),
            ("right", right)
        ]

    def process_message(self, message, data_collector):
        if len(self.controls) == 0:
            data_collector.network_interface.disconnect()
            print("bon bah on ferme")
            return
        else:
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)


def get_mlp_path(initial_position, initial_angle, initial_speed, gate_position, list_controls, ip):
    #print("addresse: ", ip, "port: ", PORT)
    nn_brain = ExampleNNMsgProcessor(list_controls)

    data_window = DataCollectionEvaluatePilot(
        nn_brain.process_message,
        address=ip,
        port=PORT,
        initial_position=initial_position,
        initial_angle=initial_angle,
        initial_speed=initial_speed,
        record=True,
        record_image=True
    )

    try:
        data_window.gate.set_gate(gate_position[0], gate_position[1], gate_position[2])

        while data_window.network_interface.is_connected():
            data_window.network_interface.recv_msg()
        return data_window.recorded_data
    except:
        print("disconnected.")
        data_window.network_interface.disconnect()
  
        
def process_json_files(base_directory):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Process the JSON data
                        list_controls = data["controls"]
                        p1, p2, width, initial_position, initial_speed, initial_angle = data["initial_context"]
                        gate_position = [p1,p2,width]
                        print("context:")
                        print(data["initial_context"])
                        print("controls: ")
                        print(list_controls)
                        recorded_data = get_mlp_path(initial_position, initial_angle, initial_speed, gate_position, list_controls,"127.0.0.1")
                        
                        record_name = f"{root}/record_%d.npz"
                        fid = 0
                        while os.path.exists(record_name % fid):
                            fid += 1

                        class ThreadedSaver(QtCore.QThread):
                            def __init__(self, path, data):
                                super().__init__()
                                self.path = path
                                self.data = data

                            def run(self):
                                with lzma.open(self.path, "wb") as f:
                                    pickle.dump(self.data, f)

                        saving_worker = ThreadedSaver(record_name % fid, recorded_data)
                        saving_worker.start()

                        
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")


base_directory = "out"  # directory with the control files
process_json_files(base_directory)