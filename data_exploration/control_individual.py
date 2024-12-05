import requests
import time
from gate import Gate
import math
import json

def send_simulation_request(protocol, server_ip, port, gate_p1, gate_p2, thickness, car_position, car_speed, car_angle, list_controls, deltaT, timeout=30):
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
    """ CONSTRUCT ENDPOINT URL FOR THE SIMULATION"""
    url = f"{protocol}://{server_ip}:{port}/simulate"
    """ DEFINE HTTP HEADER FOR THE REQUEST """
    headers = {'Content-Type': 'application/json'}
    """ PREPARE PAYLOAD TO SEND IN THE POST REQUEST """
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
        """ SEND POST REQUEST """
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        if response.status_code == 200:
            response_data = response.json() # PARSE THE JSON
            status = response_data.get("status", False)
            steps = response_data.get("steps", -1) 
            collisions = response_data.get("collisions", -1)
            response.close()
            return True, status, steps, collisions
        else:
            """ HANDLE SERVER ERROR """
            response.close()
            return False, False, -1, -1
        
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False, False, -1, -1

if __name__ == "__main__":

    """ SETUP CONTEXT AND CONTROLS """
    car_position = [  # [x, y, z]
             47.183097939402664,
            0,
            -94.90973925787584
        ]  
    car_speed = 28.318729600865794 
    car_angle = 120 
    gate_position = [[   # [gate_p1, gate_p2, gate_thickness]
            79,
            -138
        ], [
            54,
            -140
        ], 5]  
    list_controls =   [
        [
            1,
            0,
            1,
            0
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            0,
            0,
            0,
            1
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            1,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            0
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            0,
            0,
            0,
            1
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            1,
            0,
            0,
            1
        ],
        [
            1,
            0,
            1,
            0
        ],
        [
            0,
            0,
            1,
            0
        ]
    ]
    
    
    
    

    deltaT = 0.1
    reussied, status, steps, col = send_simulation_request(
        "http", 
        "127.0.0.1", 
        5000, 
        gate_position[0], 
        gate_position[1], 
        gate_position[2], 
        car_position, 
        car_speed, 
        car_angle, 
        list_controls, 
        deltaT
    )
    
    print("Succeeded?:", reussied)
    print("Simulation Status:", status)
    print("Number of deltaT steps to arrive at gate:", steps)
    print("Number of collisions:", col)