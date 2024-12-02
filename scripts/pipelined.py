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


class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        self.model = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("models/MLP_model.pth", weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def nn_infer(self, message):
        X = torch.tensor([list(message.raycast_distances) + [message.car_speed]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X)

        output_list = output.tolist()[0]
        formatted_output = ["{:.4f}".format(x) for x in output_list] 

        forward = float(formatted_output[0]) > 0.5
        backward = float(formatted_output[1]) > 0.5
        left = float(formatted_output[2]) > 0.5
        right = float(formatted_output[3]) > 0.5

        car_speed = message.car_speed

        if not forward and not backward and abs(car_speed) < 0.3:
            if car_speed < 0:
                backward = True
            else:
                forward = True

        return [
            ("forward", forward),
            ("back", backward),
            ("left", left),
            ("right", right)
        ]

    def process_message(self, message, data_collector):
        car_position = message.car_position

        if data_collector.gate.is_car_through((car_position[0], car_position[2])) and len(data_collector.recorded_data) > 2:
            data_collector.network_interface.disconnect()
            print("bon bah on ferme")
            return
        else:
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)

def get_mlp_path(initial_position, initial_angle, initial_speed, gate_position, ip):
    print("addresse: ", ip, "port: ", PORT)
    nn_brain = ExampleNNMsgProcessor()

    data_window = DataCollectionEvaluatePilot(
        nn_brain.process_message,
        address=ip,
        port=PORT,
        initial_position=initial_position,
        initial_angle=initial_angle,
        initial_speed=initial_speed,
        record=True,
        record_image=False
    )

    try:
        data_window.gate.set_gate(gate_position[0], gate_position[1], gate_position[2])

        while data_window.network_interface.is_connected():
            data_window.network_interface.recv_msg()
        return data_window.recorded_data
    except:
        print("disconnected.")
        data_collector.network_interface.disconnect()

# Load gate configurations from JSON file
with open("gates_simple_track.json", "r") as file:
    gate_configurations = json.load(file)["gate_position_simple"]

allPods = get_pods()
print(allPods)

# Process each gate
for idx, gate_config in enumerate(gate_configurations):
    for i in range(N_ITERATIONS):
        gate_output = []

        # Start parameters
        initial_position = gate_config["start_position"]
        initial_angle = gate_config["start_orientation"]
        gate_position = (
            gate_config["p1_gate"],
            gate_config["p2_gate"],
            GATE_WIDTH  # Fixed gate width
        )
        initial_speed = random.randrange(-20, 51)

        initial_context = [gate_config["p1_gate"], gate_config["p2_gate"], GATE_WIDTH, initial_position, initial_speed, initial_angle]

        # Generate multiple individuals for the gate
        for individual_idx in range(10):  # 10 individuals per gate
            # little delay for the socket to be free again
            time.sleep(0.9)
            # Generate recorded data for the individual
            recorded_data = get_mlp_path(initial_position, initial_angle, initial_speed, gate_position,"127.0.0.1")

            # Format the recorded data
            formatted_data = (
                f"individual_{idx}_{individual_idx}",
                [tuple(snapshot.current_controls) for snapshot in recorded_data]
            )
            gate_output.append(formatted_data)


        out = genAl(3,gate_output,initial_context)
        print("initial context: ")
        print(initial_context)
        print("final trajectory: ")
        print(out)
        # TODO: prints the genAl into a file with the initial context
