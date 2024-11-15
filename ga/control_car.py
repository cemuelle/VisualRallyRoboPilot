import requests
import time
import numpy as np

class CarController:
    def __init__(self, protocol, server_ip, port, gate_p1=None, gate_p2=None, gate_thickness=5):
        self.protocol = protocol
        self.server_ip = server_ip
        self.port = port
        self.commands = ["forward", "back", "left", "right"]
        # Set gate properties with default values if not provided
        self.gate_p1 = gate_p1 if gate_p1 else [0, -15]
        self.gate_p2 = gate_p2 if gate_p2 else [0, 15]
        self.gate_thickness = gate_thickness

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

    def is_point_in_gate(self, point):
        # Convert the points to numpy arrays for easier vector manipulation
        p1 = np.array(self.gate_p1)
        p2 = np.array(self.gate_p2)
        point = np.array(point)

        # Compute the direction vector of the gate segment and its length
        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length == 0:
            raise ValueError("The points p1 and p2 cannot be the same.")

        # Normalize the direction vector
        direction_normalized = direction / length

        # Compute the normal vector to the direction
        normal = np.array([-direction_normalized[1], direction_normalized[0]])

        # Compute the vectors from p1 to the point and from p2 to the point
        vector_p1_to_point = point - p1

        # Project the point onto the direction vector to see if it is between p1 and p2
        projection_length = np.dot(vector_p1_to_point, direction_normalized)
        
        # Check if the point projection lies between p1 and p2
        if projection_length < 0 or projection_length > length:
            return False

        # Compute the distance of the point from the line segment
        distance_from_line = abs(np.dot(vector_p1_to_point, normal))

        # Check if the distance is within the given thickness
        if distance_from_line <= self.gate_thickness:
            return True
        else:
            return False

if __name__ == "__main__":
    server_ip = "127.0.0.1"
    port = 5000
    protocol = "http"  # "http" or "https"

    # Initialize CarController and set gate details
    car_controller = CarController(protocol, server_ip, port)
    car_controller.set_gate([-140, -21], [-165, -24], 5)

    # Set car position, speed, and rotation
    car_controller.set_car(10, 0, 1, 50, -90)

    # Giving a list of controls to control_car
    #   Forward - Backward - Left - Right
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

    for controls in list_controls:
        car_controller.control_car(controls)
        sensing_data = car_controller.get_sensing()
        if sensing_data:
            x = sensing_data["car_position x"]
            z = sensing_data["car_position z"]
            position = (x, z)
            if car_controller.is_point_in_gate(position):
                print("Car is in the gate!")
        time.sleep(0.5)

    # while (True):
    #     # Get sensing data
    #     sensing_data = car_controller.get_sensing()
    #     if sensing_data:
    #         x = sensing_data["car_position x"]
    #         z = sensing_data["car_position z"]
    #         orientation = sensing_data["car_angle"]
    #         position = (x, z)
    #         print("Position:", position, "Orientation:", orientation)
    #         if car_controller.is_point_in_gate(position):
    #             print("Car is in the gate!")

    #     time.sleep(3)