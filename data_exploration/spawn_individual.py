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
INDIVI_NUMBER = 10
RANDOMIZATION_RANGE = 2
MAX_DURATION = 5
N_ITERATIONS = 5
GATE_WIDTH = 5
PORT = 7654
OUTPUT_DIR = "out_blue"

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
        #self.model.load_state_dict(torch.load("models/model_15.pth", weights_only=True))
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
    
    def process_message(self, message, data_collector, max_infer_time=10):
        """
        Processes a message and controls the simulation with a running elapsed time counter.
        
        Args:
            message: The input message containing car position and other details.
            data_collector: The object collecting data from the simulation.
            max_infer_time: The maximum time allowed (in seconds) since the simulation started.
        """
        car_position = message.car_position

        """ HERE WE WANT TO TIME OUT THE INDIVIDUAL WHO DIDNT COMPLETE TRAJECTORY IN TIME"""

        """ THIS AVOID KEEPING DATA OF INDIVIDUAL WHO DIDNT MAKE IT """

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
            """ EACH INDIVIDUAL WHICH DIDNT MAKE IT HAVE THERE RECORDED EMPTIED """
            data_collector.recorded_data = []
            print("Control set to 0")
            return

        """ THIS CHECK IF THE INDIVIDUAL PASSED THE GATE AND STORE THE DATA """

        """ WE AVOID IMPOSSIBLE TRAJECTORY BY SETTING A MINIMUM OF CONTROLS RECORDED"""

        # Check if the car has passed through the gate and collected enough data
        if data_collector.gate.is_car_through((car_position[0], car_position[2])):
            print(f"Car passed through the gate. Recorded data count: {len(data_collector.recorded_data)}")
            if len(data_collector.recorded_data) > 2:
                data_collector.network_interface.disconnect()
                print("Simulation disconnected.")
                return
            else:
                print("Car passed through the gate but insufficient data recorded.")

        """ THIS INFER THE COMMANDS"""

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
    """ 
    This function sets up and evaluates a neural network-based pilot.
    
    Parameters:
        initial_position (list): The starting position of the pilot as [x, y, z].
        initial_angle (float): The initial orientation angle of the pilot.
        initial_speed (float): The initial speed of the pilot.
        gate_position (list): The position of the gate to be passed as [x, y, z].
        ip (str): The IP address of the server to connect to.
    
    Returns:
        list: Recorded data collected during the evaluation.
    """

    # Log the server connection details.
    print("address: ", ip, "port: ", PORT)
    
    # Initialize the neural network message processor.
    nn_brain = ExampleNNMsgProcessor()

    # Create a DataCollectionEvaluatePilot instance for managing the simulation and data collection.
    data_window = DataCollectionEvaluatePilot(
        nn_brain.process_message,  # Neural network callback for processing messages.
        address=ip,               # Server IP address.
        port=PORT,                # Server port.
        initial_position=initial_position,  # Initial position of the pilot.
        initial_angle=initial_angle,        # Initial angle of the pilot.
        initial_speed=initial_speed,        # Initial speed of the pilot.
        record=True,             # Enable data recording.
        record_image=False       # Disable image recording to save resources.
    )

    try:
        # Set the gate position in the simulation using the provided coordinates.
        data_window.gate.set_gate(gate_position[0], gate_position[1], gate_position[2])

        # Continuously process messages while the network interface is connected.
        while data_window.network_interface.is_connected():
            data_window.network_interface.recv_msg()

        # Return the recorded data after the connection ends.
        return data_window.recorded_data
    
    except Exception as e:
        # Handle disconnection and log the error.
        print("disconnected:", e)
        data_window.network_interface.disconnect()




""" LOAD THE GATES OF A TRACK"""

with open("gatesdream.json", "r") as file:
    gate_configurations = json.load(file)["gate_positions_blue"]


""" SAVE THE RECORDED DATA INTO A JSON FORMAT WITH CONTEXT AND CONTROLS"""

def save_individual_data(directory, individual_name, initial_context, control_data):
    os.makedirs(directory, exist_ok=True)
    individual_file_path = os.path.join(directory, f"{individual_name}.json")
    with open(individual_file_path, "w") as f:
        json.dump({
            "initial_context": initial_context,
            "controls": control_data
        }, f, indent=4)



for idx, gate_config in enumerate(gate_configurations):
    """ THIS WILL ITERATE OVER THE GATES INTO THE SPECIFIC TRACK"""

    for iteration in range(N_ITERATIONS):

        """ THIS WILL CREATE A INITIAL CONTEXT SPECIFIC TO THIS ITERATION"""

        """ ALL INDIVIDUAL INSIDE THE ITERATION WILL SHARE THOSE PARAMS"""
        base_position = gate_config["start_position"]
        ## THIS RANDOMIZE THE X AND Y AXIS TO PLACE THE CAR IN DIFFERENT POSITION
        randomized_position = [
            base_position[0] + random.uniform(-RANDOMIZATION_RANGE, RANDOMIZATION_RANGE),
            base_position[1],
            base_position[2] + random.uniform(-RANDOMIZATION_RANGE, RANDOMIZATION_RANGE)
        ]

        
        initial_position = randomized_position[:]  # Make sure to copy the list
        initial_angle = gate_config["start_orientation"]
        gate_position = (
            gate_config["p1_gate"],
            gate_config["p2_gate"],
            GATE_WIDTH  
        )
        initial_speed = random.uniform(-20, 35)
        """ THIS WILL GROUP ALL ABOVE PARAMS TO CREATE THE CONTEXT """
        initial_context = [gate_config["p1_gate"], gate_config["p2_gate"], GATE_WIDTH, initial_position, initial_speed, initial_angle]


        
        # Output directory for context(gates) and iteration 
        # OUT > CONTEXT_0, CONTEXT_1 > ITERATION_0, ITERATION_1 
        context_dir = os.path.join(OUTPUT_DIR, f"context_{idx}")
        iteration_dir = os.path.join(context_dir, f"iteration_{iteration}")

        """ THIS ENSURE WE HAVE ENOUGH SAVED INDIVIDUAL PER ITERATION"""
        saved_individuals = 0



        # Generate multiple individuals for the gate
        """ GENERATE INDIVIDUAL UNTIL WE HAVE ENOUGH INDIVI_NUMBER """
        while saved_individuals < INDIVI_NUMBER:  
            time.sleep(0.9)  # Delay for the socket to be free again

            """ ENSURE WE HAVE THE SAME POSITION FOR EACH INDIVIDUAL """
            #COULD BE OVERKILL
            position_to_use = initial_position[:]  


            # SIMULATE AND STOCK THE DATA 
            recorded_data = get_mlp_path(position_to_use, initial_angle, initial_speed, gate_position, "127.0.0.1")

            # Print position to verify consistency
            #print(f"Position during individual {saved_individuals} generation:", position_to_use)

            # REFORMAT THE RECORDED DATA
            control_data = [tuple(snapshot.current_controls) for snapshot in recorded_data]
            """ DATA IS SAVED INTO THE JSON FOR THE INDIVIDUAL """
            """IF THE RECORDED DATA IS EMPTY, IT MEANS IT IS AN INDIVIDUAL WHICH WAS TIMEOUT THUS DATA WAS EMPTIED"""
            
            if control_data:
                individual_name = f"individual_{saved_individuals}"
                save_individual_data(iteration_dir, individual_name, initial_context, control_data)
                saved_individuals += 1
            else:
                print(f"Skipping saving for individual {saved_individuals} due to empty control data.")



        
        
