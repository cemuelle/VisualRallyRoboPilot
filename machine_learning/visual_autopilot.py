import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets
from machine_learning.models import AlexNetPerso

from data_collector import DataCollectionUI
"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


model_path = "../rallyBot/models/augmented_nn.pth"
model_dict = torch.load(model_path)

model = AlexNetPerso(4, 0)
model.load_state_dict(model_dict)
model.eval()

output_feature_labels = ['forward', 'backward', 'left', 'right', 'nothing']

def preprocess_input(s):
    # return torch.tensor([s.raycast_distances[3],
    # s.raycast_distances[4],
    # s.raycast_distances[5],
    # s.raycast_distances[6],
    # s.raycast_distances[7],
    # s.raycast_distances[8],
    # s.raycast_distances[9],
    # s.raycast_distances[10],
    # s.raycast_distances[11]], dtype=torch.float32)
    return


class VisualNNMsgProcessor:
    def __init__(self):
        self.model = model

    def nn_infer(self, message):
        input_tensor = preprocess_input(message)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if necessary

        with torch.no_grad():
            output = self.model(input_tensor)
        print("Models output: ", output)

        output_list = output.tolist()[0]
        formatted_output = ["{:.4f}".format(x) for x in output_list] 

        # convert the output to boolean values
        forward = float(formatted_output[0]) > 0.5
        backward = float(formatted_output[1]) > 0.5
        left = float(formatted_output[2]) > 0.5
        right = float(formatted_output[3]) > 0.5

        # make sure the car is not moving forward and backward at the same time
        if forward and backward:
            forward = formatted_output[0] > formatted_output[1]
            backward = not forward
        if left and right:
            left = formatted_output[2] > formatted_output[3]
            right = not left

        print(f"forward {formatted_output[0]} is {forward}")
        print(f"back {formatted_output[1]} is {backward}")
        print(f"left {formatted_output[2]} is {left}")
        print(f"right {formatted_output[3]} is {right}")

        return [
            ("forward", forward),
            ("back", backward),
            ("left", left),
            ("right", right)
        ]
        

    def process_message(self, message, data_collector):

        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = VisualNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
