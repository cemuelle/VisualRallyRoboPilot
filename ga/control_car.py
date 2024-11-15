import requests
import time

class CarController:
    def __init__(self, protocol, server_ip, port):
        self.protocol = protocol
        self.server_ip = server_ip
        self.port = port
        self.commands = ["forward", "back", "left", "right"]

    def send_command(self, command):
        url = f"{self.protocol}://{self.server_ip}:{self.port}/command"
        headers = {'Content-Type': 'application/json'}
        data = {
            "command": command
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                print("Success:", response.json())
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

if __name__ == "__main__":
    server_ip = "127.0.0.1"  # Replace with your server's IP
    port = 5000  # Replace with your desired port
    protocol = "http"  # Replace with "http" or "https" as needed

    car_controller = CarController(protocol, server_ip, port)

    # Sending different commands
    car_controller.send_command("set position 10,0,1;")
    car_controller.send_command("set speed 50;")
    car_controller.send_command("reset;")

    # Giving a list of controls to control_car
    list_controls = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],  
        [0, 0, 1, 0],
        [1, 0, 1, 0],  
        [1, 0, 1, 0],
        [1, 0, 0, 1], 
        [0, 0, 0, 0] 
    ]

    for controls in list_controls:
        car_controller.control_car(controls)
        # Uncomment to get sensing data
        # sensing_data = car_controller.get_sensing()
        # if sensing_data:
        #     print("Sensing Data:", sensing_data)
        time.sleep(0.5)