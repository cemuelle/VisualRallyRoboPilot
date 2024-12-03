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
#from ga_pipeline import genAl, get_pods
#from kubernetes import client, config

#config.load_kube_config()

#variable
onlyOnce = True

# parameters
INDIVI_NUMBER = 2
RANDOMIZATION_RANGE = 8
MAX_DURATION = 5
N_ITERATIONS = 3
GATE_WIDTH = 5
PORT = 7654
OUTPUT_DIR = "out"

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
        #self.model.load_state_dict(torch.load("models/MLP_model.pth", weights_only=True))
        self.model.load_state_dict(torch.load("models/model_15.pth", weights_only=True))
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

        #if not forward and not backward and abs(car_speed) < 0.3:
        if not forward and not backward and abs(car_speed) < 2:
            if car_speed < 0:
                backward = True
            else:
                forward = True

        if forward and backward:
            forward = formatted_output[0] > formatted_output[1]
            backward = not forward
        if left and right:
            left = formatted_output[2] > formatted_output[3]
            right = not left
 
        # # make sure the car isn't moving too fast
        if car_speed > 20:
            forward = False
            backward = True
 
        # if message.raycast_distances[7] < 50 and car_speed > 30:
        #     forward = False
        #     backward = True
        if message.raycast_distances[7] > 10 and car_speed < -2:
            forward = True
            backward = False

        return [
            ("forward", forward),
            ("back", backward),
            ("left", left),
            ("right", right)
        ]

    

    

    def process_message(self, message, data_collector, max_infer_time=30):
        """
        Processes a message and controls the simulation with a running elapsed time counter.
        
        Args:
            message: The input message containing car position and other details.
            data_collector: The object collecting data from the simulation.
            max_infer_time: The maximum time allowed (in seconds) since the simulation started.
        """
        car_position = message.car_position

        # Initialize start time if not already set
        if not hasattr(data_collector, 'start_time'):
            data_collector.start_time = time.perf_counter()

        # Calculate the total elapsed time since the start
        elapsed_time = time.perf_counter() - data_collector.start_time
        #print(f"Total elapsed time since start: {elapsed_time:.2f} seconds")

        # Check if elapsed time exceeds the maximum allowed
        if elapsed_time > max_infer_time:
            print(f"Elapsed time exceeded {max_infer_time} seconds. Restarting simulation.")
            data_collector.network_interface.disconnect()
            print("Simulation disconnected due to timeout.")
            data_collector.recorded_data = []
            print("Control set to 0")
            return

        # Check if the car has passed through the gate and collected enough data
        if data_collector.gate.is_car_through((car_position[0], car_position[2])):
            print(f"Car passed through the gate. Recorded data count: {len(data_collector.recorded_data)}")
            if len(data_collector.recorded_data) > 2:
                data_collector.network_interface.disconnect()
                print("Simulation disconnected.")
                return
            else:
                print("Car passed through the gate but insufficient data recorded.")

        # Perform inference
        try:
            commands = self.nn_infer(message)
        except Exception as e:
            data_collector.network_interface.disconnect()
            return

        # Apply the commands
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
        data_window.network_interface.disconnect()

# Load gate configurations from JSON file
with open("gatesSH.json", "r") as file:
    gate_configurations = json.load(file)["gate_slightly_harder"]

#allPods = get_pods()
#print(allPods)

# Function to save individual data as JSON
def save_individual_data(directory, individual_name, initial_context, control_data):
    os.makedirs(directory, exist_ok=True)
    individual_file_path = os.path.join(directory, f"{individual_name}.json")
    with open(individual_file_path, "w") as f:
        json.dump({
            "initial_context": initial_context,
            "controls": control_data
        }, f, indent=4)

for idx, gate_config in enumerate(gate_configurations):
    for iteration in range(N_ITERATIONS):
        # Prepare initial context
        base_position = gate_config["start_position"]
        randomized_position = [
            base_position[0] + random.uniform(-RANDOMIZATION_RANGE, RANDOMIZATION_RANGE),
            base_position[1],
            base_position[2] + random.uniform(-RANDOMIZATION_RANGE, RANDOMIZATION_RANGE)
        ]

        # Assign the same randomized position for the whole iteration
        initial_position = randomized_position[:]  # Make sure to copy the list
        initial_angle = gate_config["start_orientation"]
        gate_position = (
            gate_config["p1_gate"],
            gate_config["p2_gate"],
            GATE_WIDTH  # Fixed gate width
        )
        initial_speed = random.randrange(-20, 51)
        initial_context = [gate_config["p1_gate"], gate_config["p2_gate"], GATE_WIDTH, initial_position, initial_speed, initial_angle]

        # Directory for the current gate context and iteration
        context_dir = os.path.join(OUTPUT_DIR, f"context_{idx}")
        iteration_dir = os.path.join(context_dir, f"iteration_{iteration}")
        saved_individuals = 0
        # Generate multiple individuals for the gate
        while saved_individuals < INDIVI_NUMBER:  # Adjusted for 10 individuals per gate
            time.sleep(0.9)  # Delay for the socket to be free again

            # Verify that initial_position is not modified
            position_to_use = initial_position[:]  # Always work with a copy
            # Simulate data retrieval (replace with actual function)
            recorded_data = get_mlp_path(position_to_use, initial_angle, initial_speed, gate_position, "127.0.0.1")

            # Print position to verify consistency
            print(f"Position during individual {saved_individuals} generation:", position_to_use)

            # Format control data
            control_data = [tuple(snapshot.current_controls) for snapshot in recorded_data]

            if control_data:
                individual_name = f"individual_{saved_individuals}"
                save_individual_data(iteration_dir, individual_name, initial_context, control_data)
                saved_individuals += 1
            else:
                print(f"Skipping saving for individual {saved_individuals} due to empty control data.")



        
        
