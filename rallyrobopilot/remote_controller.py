
from ursina import *
import socket
import numpy as np
from ga.gate import Gate
import time

from flask import Flask, request, jsonify


from .sensing_message import SensingSnapshot, SensingSnapshotManager
from .remote_commands import RemoteCommandParser


REMOTE_CONTROLLER_VERBOSE = False
is_ready = True


def printv(str):
    if REMOTE_CONTROLLER_VERBOSE:
        print(str)

class RemoteController(Entity):
    def __init__(self, car = None, connection_port = 7654, flask_app=None):
        super().__init__()

        self.last_time= -1

        self.ip_address = "127.0.0.1"
        self.port = connection_port
        self.car = car

        self.listen_socket = None
        self.connected_client = None

        self.client_commands = RemoteCommandParser()

        self.list_controls = []
        self.start_simulate_controls = False
        self.pass_gate = False
        self.step = 0
        self.gate = Gate()


        # Setup http route for updating.
        @flask_app.route('/command', methods=['POST'])
        def send_command_route():
            if self.car is None:
                return jsonify({"error": "No car connected"}), 400

            command_data = request.json
            if not command_data or 'command' not in command_data:
                return jsonify({"error": "Invalid command data"}), 400

            try:
                self.client_commands.add(command_data['command'].encode())
                return jsonify({"status": f"Command received: {command_data['command']}"}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            

        @flask_app.route('/simulate', methods=['POST'])
        def simulate_route():
            global is_ready

            if not is_ready:
                return jsonify({"error": "Pod in use"}), 503

            is_ready = False
            if self.car is None:
                is_ready = True
                return jsonify({"error": "No car connected"}), 400

            command_data = request.json
            
            # Validate the presence and format of each required parameter
            required_fields = {
                "gate_p1": list,
                "gate_p2": list,
                "thickness": (int, float),
                "car_position": list,
                "car_speed": (int, float),
                "car_angle": (int, float),
                "list_controls": list
            }

            # Check if all required fields are present and have the correct type
            for field, expected_type in required_fields.items():
                if field not in command_data:
                    is_ready = True
                    return jsonify({"error": f"Missing required field: {field}"}), 400
                if not isinstance(command_data[field], expected_type):
                    is_ready = True
                    return jsonify({"error": f"Invalid type for field: {field}. Expected {expected_type}"}), 400
                
            # Additional checks for specific fields
            if len(command_data['gate_p1']) != 2 or not all(isinstance(x, (int, float)) for x in command_data['gate_p1']):
                is_ready = True
                return jsonify({"error": "Invalid format for gate_p1. Expected a list of two numbers."}), 400

            if len(command_data['gate_p2']) != 2 or not all(isinstance(x, (int, float)) for x in command_data['gate_p2']):
                is_ready = True
                return jsonify({"error": "Invalid format for gate_p2. Expected a list of two numbers."}), 400

            if len(command_data['car_position']) != 3 or not all(isinstance(x, (int, float)) for x in command_data['car_position']):
                is_ready = True
                return jsonify({"error": "Invalid format for car_position. Expected a list of three numbers."}), 400

            if not isinstance(command_data['list_controls'], list) or len(command_data['list_controls']) == 0 or not all(
                    isinstance(control, list) and len(control) == 4 and all(isinstance(x, (int, float)) for x in control)
                    for control in command_data['list_controls']
            ):
                is_ready = True
                return jsonify({"error": "Invalid format for list_controls. Expected a non-empty list of lists, each containing four numbers."}), 400
        

            try:
                self.start_simulate_controls = True
                self.pass_gate = False

                self.gate.set_gate(command_data['gate_p1'], command_data['gate_p2'], command_data['thickness'])
                # define car position, speed and rotation and then reset the car
                self.step = 0
                self.car.reset_position = command_data['car_position']
                self.car.reset_speed = command_data['car_speed']
                self.car.reset_orientation = (0, command_data['car_angle'], 0)
                self.car.reset_car()

                self.list_controls = command_data['list_controls']

                while self.start_simulate_controls:
                    time.sleep(0.1)

                if self.pass_gate:
                    is_ready = True
                    return jsonify({"status": True, "steps": self.step, "collisions": self.car.collisions}), 200
                else:
                    is_ready = True
                    return jsonify({"status": False, "steps": -1, "collisions": self.car.collisions}), 200
            except Exception as e:
                is_ready = True
                return jsonify({"error": str(e)}), 500
    
        @flask_app.route('/sensing')
        def get_sensing_route():
            return jsonify(self.get_sensing_data()), 200


    def update(self):
        if not self.start_simulate_controls:
            self.update_network()
            self.process_remote_commands()
        else :
            self.process_simulate_controls()
        self.process_sensing()
            

    def process_simulate_controls(self):
        if self.car is None:
            return
        
        car_position = self.car.world_position
        if self.gate.is_car_through((car_position[0], car_position[2])):
            self.pass_gate = True
            self.start_simulate_controls = False
            held_keys['w'] = False
            held_keys['s'] = False
            held_keys['a'] = False
            held_keys['d'] = False
            return
        len_controls = len(self.list_controls)
        if len_controls > 0:
            controls = self.list_controls.pop(0)
            self.step += 1

            held_keys['w'] = controls[0] == 1
            held_keys['s'] = controls[1] == 1
            held_keys['a'] = controls[2] == 1
            held_keys['d'] = controls[3] == 1
        else:
            held_keys['w'] = False
            held_keys['s'] = False
            held_keys['a'] = False
            held_keys['d'] = False
            self.start_simulate_controls = False


    def process_sensing(self):
        if self.car is None or self.connected_client is None:
            return

        snapshot = SensingSnapshot()
        snapshot.current_controls = (held_keys['w'] or held_keys["up arrow"],
                                     held_keys['s'] or held_keys["down arrow"],
                                     held_keys['a'] or held_keys["left arrow"],
                                     held_keys['d'] or held_keys["right arrow"])
        snapshot.car_position = self.car.world_position
        snapshot.car_speed = self.car.speed
        snapshot.car_angle = self.car.rotation_y
        snapshot.raycast_distances = self.car.multiray_sensor.collect_sensor_values()

        #   Collect last rendered image
        tex = base.win.getDisplayRegion(0).getScreenshot()
        arr = tex.getRamImageAs("RGB")
        data = np.frombuffer(arr, np.uint8)
        image = data.reshape(tex.getYSize(), tex.getXSize(), 3)
        image = image[::-1, :, :]#   Image arrives with inverted Y axis

        snapshot.image = image

        msg_mngr = SensingSnapshotManager()
        data = msg_mngr.pack(snapshot)

        self.connected_client.settimeout(0.01)
        try:
            self.connected_client.sendall(data)
        except socket.error as e:
            print(f"Socket error: {e}")
            self.disconnect()

    def get_sensing_data(self):
        current_controls = (held_keys['w'] or held_keys["up arrow"],
                            held_keys['s'] or held_keys["down arrow"],
                            held_keys['a'] or held_keys["left arrow"],
                            held_keys['d'] or held_keys["right arrow"])
        car_position = self.car.world_position
        car_speed = self.car.speed
        car_angle = self.car.rotation_y
        raycast_distances = self.car.multiray_sensor.collect_sensor_values()
        return {'up': current_controls[0],
                'down': current_controls[1],
                'left': current_controls[2], 
                'right': current_controls[3],
                'car_position x': car_position[0],
                'car_position y': car_position[1],
                'car_position z': car_position[2],
                'car_speed': car_speed,
                'car_angle': car_angle,
                'raycast_distances 0': raycast_distances[0],
                'raycast_distances 1': raycast_distances[1],
                'raycast_distances 2': raycast_distances[2],
                'raycast_distances 3': raycast_distances[3],
                'raycast_distances 4': raycast_distances[4],
                'raycast_distances 5': raycast_distances[5],
                'raycast_distances 6': raycast_distances[6],
                'raycast_distances 7': raycast_distances[7],
                'raycast_distances 8': raycast_distances[8],
                'raycast_distances 9': raycast_distances[9],
                'raycast_distances 10': raycast_distances[10],
                'raycast_distances 11': raycast_distances[11],
                'raycast_distances 12': raycast_distances[12],
                'raycast_distances 13': raycast_distances[13],
                'raycast_distances 14': raycast_distances[14]
                }

    def process_remote_commands(self):
        if self.car is None:
            return

        while len(self.client_commands) > 0:
            try:
                commands = self.client_commands.parse_next_command()
                print("Processing command", commands)
                if commands[0] == b'push' or commands[0] == b'release':
                    if commands[1] == b'forward':
                        held_keys['w'] = commands[0] == b'push'
                    elif commands[1] == b'back':
                        held_keys['s'] = commands[0] == b'push'
                    elif commands[1] == b'right':
                        held_keys['d'] = commands[0] == b'push'
                    elif commands[1] == b'left':
                        held_keys['a'] = commands[0] == b'push'
                              
                # Release all
                if commands[0] == b'release' and commands[1] == b'all':
                    print("received release all command")
                    held_keys['w'] = False
                    held_keys['s'] = False
                    held_keys['d'] = False
                    held_keys['a'] = False

                elif commands[0] == b'set':
                    if commands[1] == b'position':
                        self.car.reset_position = commands[2]
                    elif commands[1] == b'rotation':
                        self.car.reset_orientation = (0, commands[2], 0)
                    elif commands[1] == b'speed':        
                        self.car.reset_speed = float(commands[2])

                    elif commands[1] == b'ray':
                        self.car.multiray_sensor.set_enabled_rays(commands[2] == b'visible')

                elif commands[0] == b'reset':
                    self.car.reset_car()

                elif commands[0] == b'reset_collisions':
                    self.car.collisions = 0

            #   Error is thrown when commands do not fit the model --> disconnect client
            except Exception as e:
                print("Invalid command --> disconnecting : " + str(e))
                self.disconnect()

    def update_network(self):
        if self.connected_client is not None:
            data = []
            try:
                while True:
                    recv_data = self.connected_client.recv(1024)

                    #received nothing
                    if len(recv_data) == 0:
                        break
                    self.client_commands.add(recv_data)

            except socket.timeout:
                pass
            except Exception as e:
                printv(e)
                self.disconnect()

        #   No controller connected
        else:
            if self.listen_socket is None:
                self.open_connection_socket()
            try:
                inc_client, address = self.listen_socket.accept()
                print("Controller connecting from " + str(address))
                self.connected_client = inc_client
                # self.connected_client.setblocking(False)
                self.connected_client.settimeout(0.01)

                #   Close listen socket
                self.listen_socket.close()
                self.listen_socket = None
            except Exception as e:
                printv(e)


    def open_connection_socket(self):
        print("Waiting for connections")
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.bind((self.ip_address, self.port))
        # self.listen_socket.setblocking(False)
        self.listen_socket.settimeout(0.01)
        self.listen_socket.listen()

    def disconnect(self):
        if self.connected_client:
            self.connected_client.close()
            self.connected_client = None
        self.open_connection_socket()
        self.waitCommand = False
