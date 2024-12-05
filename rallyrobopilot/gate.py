import numpy as np

"""
This class represents a gate that the car has to pass through.
"""
class Gate:
    """
    Initializes the gate with the given parameters.
    gate_p1: The first point of the gate segment.
    gate_p2: The second point of the gate segment.
    thickness: The thickness of the gate.
    """
    def __init__(self, gate_p1=None, gate_p2=None, thickness=5):
        self.gate_p1 = gate_p1
        self.gate_p2 = gate_p2
        self.thickness = thickness

    """
    Sets the gate with the given parameters.
    gate_p1: The first point of the gate segment.
    gate_p2: The second point of the gate segment.
    thickness: The thickness of the gate.
    """
    def set_gate(self, gate_p1, gate_p2, thickness):
        self.gate_p1 = gate_p1
        self.gate_p2 = gate_p2
        self.thickness = thickness

    """
    Checks if the car is through the gate.
    car_position: The position of the car.
    """
    def is_car_through(self, car_position):
        p1 = np.array(self.gate_p1)
        p2 = np.array(self.gate_p2)
        point = np.array(car_position)

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
        return distance_from_line <= self.thickness
    
class GateSequence:
    """
    Initializes the gate sequence with the given parameters.
    gates: The list of gates in the sequence.
    """
    def __init__(self, gates=None):
        self.gates = gates if gates else []
        self.current_gate = 0

    """
    Adds a gate to the sequence.
    gate: The gate to add.
    """
    def add_gate(self, gate):
        self.gates.append(gate)

    """
    Adds multiple gates to the sequence.
    gates: The list of gates to add.
    """
    def add_gates(self, gates):
        self.gates.extend(gates)

    """
    Gets the next gate to pass.
    """
    def get_current_gate(self):
        return self.gates[self.current_gate]
    
    """
    Gets the index of the current gate.
    """
    def get_current_gate_index(self):
        return self.current_gate
    
    """
    Gets the number of gates in the sequence.
    """
    def get_number_of_gates(self):
        return len(self.gates)
    
    """
    Resets the gate sequence.
    """
    def reset(self):
        self.gates = []
        current_gate = 0
    
    """
    Restarts the gate sequence.
    """
    def restart(self):
        self.current_gate = 0

    """
    Checks if the car has passed the current gate in the sequence.
    car_position: The position of the car.
    """
    def check_pass_gate(self, car_position):
        if self.has_crossed_all_gates():
            return True
        
        current_gate = self.gates[self.current_gate]
        if current_gate.is_car_through(car_position):
            self.current_gate += 1
            return True
        return False

    """
    Checks if the car has crossed all the gates in the sequence.
    """
    def has_crossed_all_gates(self):
        return self.current_gate == len(self.gates)
