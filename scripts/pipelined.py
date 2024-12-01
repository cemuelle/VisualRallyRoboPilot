import os
import pickle
import lzma
import glob
import numpy as np
import json
import random
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from data_collector import DataCollectionUI
from data_collector_evaluate_pilot import DataCollectionEvaluatePilot
from PyQt6 import QtWidgets
import sys
import torch
import torch.nn as nn
import json
from ga.ga_pipeline import genAl

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
            # data_collector.saveRecord(close_after_save=True)
            data_collector.network_interface.disconnect()
            QtWidgets.QApplication.quit()
        else:
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)

def get_mlp_path(initial_position, initial_angle, initial_speed, gate_position):
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionEvaluatePilot(
        nn_brain.process_message,
        initial_position=initial_position,
        initial_angle=initial_angle,
        initial_speed=initial_speed,
        record=True,
        record_image=False
    )
    data_window.gate.set_gate(gate_position[0], gate_position[1], gate_position[2])
    app.exec()

    return data_window.recorded_data


# Load gate configurations from JSON file
with open("gates_simple_track.json", "r") as file:
    gate_configurations = json.load(file)["gate_position_simple"]

# Predefined variables for gates
gate_0_output = []
gate_1_output = []
gate_2_output = [] 
gate_3_output = []  
gate_4_output = []  
gate_5_output = []  


# Mapping gate index to variable
gate_outputs = {
    0: gate_0_output,
    1: gate_1_output,
    2: gate_2_output,
    3: gate_3_output,
    4: gate_4_output,
    5: gate_5_output,
}

# Process each gate
for idx, gate_config in enumerate(gate_configurations):
    gate_output = gate_outputs[idx]
    
    # Shared parameters
    initial_position = gate_config["start_position"]
    initial_angle = gate_config["start_orientation"]
    gate_position = (
        gate_config["p1_gate"],
        gate_config["p2_gate"],
        5  # Fixed gate width
    )
    
    # Generate multiple individuals for the gate
    for individual_idx in range(10):  # 10 individuals per gate
        initial_speed = 30  # Example variation in initial speed
        
        # Generate recorded data for the individual
        recorded_data = get_mlp_path(initial_position, initial_angle, initial_speed, gate_position)
        
        # Format the recorded data
        formatted_data = (
            f"individual_{idx}_{individual_idx}",
            [tuple(snapshot.current_controls) for snapshot in recorded_data]
        )
        gate_output.append(formatted_data)


'''
    gate_p1 = [-140,-21]
    gate_p2 = [-165,-24]
    thickness = 5
    car_position = [10,0,1]  # x, y, z
    car_speed = 50
    car_angle = -90
'''
params = [gate_config["p1_gate"],gate_config["p2_gate"],5,gate_config["start_position"],30,gate_config["start_orientation"]]
genAl(3,gate_0_output,params)

#genetic_algorithm(generation=3, mutation_rate=0, population_size=10, elitism_count=1, individual_controls=gate_0_output)
#genetic_algorithm(generation=3, mutation_rate=0, population_size=10, elitism_count=1, individual_controls=gate_1_output)
#genetic_algorithm(generation=3, mutation_rate=0, population_size=10, elitism_count=1, individual_controls=gate_2_output)







