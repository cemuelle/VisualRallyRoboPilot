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
        self.gate_p1 = gate_p1 if gate_p1 else [0, -15]
        self.gate_p2 = gate_p2 if gate_p2 else [0, 15]
        self.gate_thickness = thickness

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