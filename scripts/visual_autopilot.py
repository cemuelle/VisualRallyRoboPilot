

import sys
import torch
import torch.nn as nn
from PyQt6 import QtWidgets
from machine_learning.models import AlexNetPerso
from machine_learning.preprocessing import preprocess

from data_collector import DataCollectionUI

from machine_learning.utils import colToRgb
from rallyrobopilot import *
from PIL import Image
r"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


model_path = "model_20241112_132920_7.pth"
model_dict = torch.load(model_path, weights_only=True)

model = AlexNetPerso(4)
model.load_state_dict(model_dict)
model.eval()

output_feature_labels = ['forward', 'backward', 'left', 'right', 'nothing']


class VisualNNMsgProcessor:
    def __init__(self):
        self.model = model

    def nn_infer(self, message):
        car_speed = message.car_speed
        print(f"Car speed: {car_speed}")

        image = message.image
        image = Image.fromarray(image)
        color = colToRgb("cyan")

        image = preprocess(image)

        with torch.no_grad():
            output = self.model(image, color)
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
