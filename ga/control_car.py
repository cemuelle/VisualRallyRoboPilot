import requests
import time
from gate import Gate
import math

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
    car_controller = CarController("http", "127.0.0.1", 5000)
    result = simulate_car_movement(car_position, gate_position, list_controls, car_controller)
    print("Result:", result)



