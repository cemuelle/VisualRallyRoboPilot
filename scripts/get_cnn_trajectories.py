import requests
import time
from gate import Gate
import math
import json
import random
import os

# parameters
INDIVI_NUMBER = 10 # number of individuals to generate for each population
RANDOMIZATION_RANGE = 8 # size of the random box of the car initial position
N_ITERATIONS = 3 # number of iterations per context
GATE_WIDTH = 5 # width of the gate (finish line)
PORT = 7654 # port to use for the connexion to the main
OUTPUT_DIR = "testout" # output directory


class CarController:
    def __init__(self, protocol, server_ip, port):
        self.protocol = protocol
        self.server_ip = server_ip
        self.port = port
        self.commands = ["forward", "back", "left", "right"]

    def set_gate(self, gate_p1, gate_p2, gate_thickness):
        self.gate_p1 = gate_p1
        self.gate_p2 = gate_p2
        self.gate_thickness = gate_thickness

    def set_car(self, x, y, z, speed, rotation):
        self.send_command(f"set position {x},{y},{z};")
        self.send_command(f"set speed {speed};")
        self.send_command(f"set rotation {rotation};")
        self.send_command("reset;")

    def send_command(self, command):
        url = f"{self.protocol}://{self.server_ip}:{self.port}/command"
        headers = {'Content-Type': 'application/json','Connexion': 'close'}
        data = {
            "command": command
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print("Success:", response.json())
                pass
            else:
                print(f"Error: {response.status_code}, Response: {response.text}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")

    def get_sensing(self):
        url = f"{self.protocol}://{self.server_ip}:{self.port}/sensing"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}, Response: {response.text}")
                return None
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def control_car(self, current_controls):
        for i, control in enumerate(current_controls):
            if control == 1:
                self.send_command(f"push {self.commands[i]};")
            else:
                self.send_command(f"release {self.commands[i]};")

    


def send_simulation_request(protocol, server_ip, port, gate_p1, gate_p2, thickness, car_position, car_speed, car_angle, list_controls, deltaT=0.1, timeout=30):
    """
    Send a simulation request to the server and calculate the number of deltaT intervals 
    required for the car to arrive at the gate.

    Args:
        protocol (str): The protocol to use (e.g., "http" or "https").
        server_ip (str): The IP address of the server.
        port (int): The port number of the server.
        gate_p1 (list of float): The first gate position as a list of two floats [x, y].
        gate_p2 (list of float): The second gate position as a list of two floats [x, y].
        thickness (float): The thickness of the gate.
        car_position (list of float): The car position as a list of three floats [x, y, z].
        car_speed (float): The speed of the car.
        car_angle (float): The angle of the car.
        list_controls (list of list of float): The list of control commands, each command is a list of four floats.
        deltaT (float, optional): The time step for each control command in seconds. Defaults to 0.1.
        timeout (int, optional): The timeout for the POST request in seconds. Defaults to 30.

    Returns:
        tuple: A tuple containing:
            - status (bool): The status of the simulation (True if successful, False otherwise).
            - steps (int): The number of deltaT intervals to reach the gate or -1 if an error occurred.
    """

    url = f"{protocol}://{server_ip}:{port}/simulate"
    headers = {'Content-Type': 'application/json'}

    data = {
        "gate_p1": gate_p1,
        "gate_p2": gate_p2,
        "thickness": thickness,
        "car_position": car_position,
        "car_speed": car_speed,
        "car_angle": car_angle,
        "list_controls": list_controls,
        "deltaT": deltaT
    }

    try:
        while(True):
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
            if response.status_code == 200:
                response_data = response.json()
                response.close()
                return response_data
            elif response.status_code == 405:
                response.close()
                return(405)
            else:
                print(f"Error: {response.status_code}, Response: {response.text}")
                response.close()
                return []
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False, False, -1, -1

if __name__ == "__main__":
    # Load gate configurations from JSON file
    with open("gates_simple_track.json", "r") as file:
        gate_configurations = json.load(file)["gate_position_simple"]

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
            # Simulate data retrieval 
            deltaT = 10
            success = False
            while not success:
                recorded_data = send_simulation_request(
                    "http", 
                    "127.0.0.1", 
                    5000, 
                    gate_position[0], 
                    gate_position[1], 
                    gate_position[2], 
                    position_to_use, 
                    initial_speed, 
                    initial_angle, 
                    [[0,0,0,0],[0,0,0,0]], 
                    deltaT
                )
                if recorded_data != 405:
                    success = True

            data_without_trues = []
            try:
                for i in recorded_data:
                    i_hate_snapshots = []
                    for j in i:
                        if j:
                            i_hate_snapshots.append(1)
                        else:
                            i_hate_snapshots.append(0)
                    data_without_trues.append(i_hate_snapshots)
                    print(data_without_trues)
                # Print position to verify consistency
                print(f"Position during individual {saved_individuals} generation:", position_to_use)
            except Exception as e:
                print(str(e), " . Moving along with the procedure")

            # Format control data
            control_data = [tuple(snapshot) for snapshot in data_without_trues]

            if control_data:
                individual_name = f"individual_{saved_individuals}"
                save_individual_data(iteration_dir, individual_name, initial_context, control_data)
                saved_individuals += 1
            else:
                print(f"Skipping saving for individual {saved_individuals} due to empty control data.")
                

