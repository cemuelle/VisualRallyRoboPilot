import requests
import time
from gate import Gate
import math
import json

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
        headers = {'Content-Type': 'application/json'}
        data = {
            "command": command
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                # print("Success:", response.json())
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
       
def simulate_car_movement(car_position, gate_position, list_controls, car_controller):
    gate = Gate()
    gate.set_gate(gate_position[0], gate_position[1], gate_position[2])

    car_controller.set_car(car_position[0], car_position[1], car_position[2], car_position[3], car_position[4])

    start_time = time.time()
    crossed_gate = False

    for controls in list_controls:
        car_controller.control_car(controls)
        sensing_data = car_controller.get_sensing()
        if sensing_data:
            x = sensing_data["car_position x"]
            z = sensing_data["car_position z"]
            position = (x, z)
            if gate.is_car_through(position):
                end_time = time.time()
                crossed_gate = True
                break
        time.sleep(0.5)

    if not crossed_gate:
        return math.inf
    else:
        elapsed_time = end_time - start_time
        return elapsed_time
    
def old_main():
    car_position = [10, 0, 1, 50, -90]  # [x, y, z, speed, rotation]
    gate_position = [[-140, -21], [-165, -24], 5]  # [gate_p1, gate_p2, gate_thickness]
    list_controls = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],  
        [0, 0, 1, 0],
        [0, 1, 1, 0],  
        [1, 0, 0, 0],
        [1, 0, 0, 1], 
        [0, 0, 0, 1],
        [1, 0, 0, 1], 
        [0, 1, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
    ]
    car_controller = CarController("http", "127.0.0.1", 5000)
    result = simulate_car_movement(car_position, gate_position, list_controls, car_controller)
    print("Result:", result)


def send_simulation_request(protocol, server_ip, port, gate_p1, gate_p2, thickness, car_position, car_speed, car_angle, list_controls, timeout=30):
    """
    Send a simulation request to the server.

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
        timeout (int, optional): The timeout for the POST request in seconds. Defaults to 30.

    Returns:
        tuple: A tuple containing:
            - status (bool): The status of the simulation (True if successful, False otherwise).
            - time (float): The time taken for the simulation or float('inf') if there was an error.
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
        "list_controls": list_controls
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        if response.status_code == 200:
            response_data = response.json()
            status = response_data.get("status", False)
            time = response_data.get("time", float("inf"))
            print("Simulation started successfully:", response.json())
            return status, time
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            return False, float("inf")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    
if __name__ == "__main__":
    car_position = [10, 0, 1, 50, -90]  # [x, y, z, speed, rotation]
    gate_position = [[-140, -21], [-165, -24], 5]  # [gate_p1, gate_p2, gate_thickness]
    list_controls = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],  
        [0, 0, 1, 0],
        [0, 1, 1, 0],  
        [1, 0, 0, 0],
        [1, 0, 0, 1], 
        [0, 0, 0, 1],
        [1, 0, 0, 1], 
        [0, 1, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
    ]

    status, time = send_simulation_request("http", "127.0.0.1", 5000, gate_position[0], gate_position[1], gate_position[2], [car_position[0], car_position[1], car_position[2]], car_position[3], car_position[4], list_controls)

    print("Simulation Status:", status)
    print("Simulation Time:", time)
