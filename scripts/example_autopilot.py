from PyQt6 import QtWidgets

from data_collector import DataCollectionUI
from data_collector_evaluate_pilot import DataCollectionEvaluatePilot

"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(self.sigmoid(self.fc1(x)))
        x = self.dropout(self.sigmoid(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        self.model = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("models/MLP_model.pth", weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def nn_infer(self, message):
        X = torch.tensor([list(message.raycast_distances) + [message.car_speed]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X)

        output_list = output.tolist()[0]
        formatted_output = ["{:.4f}".format(x) for x in output_list] 

        forward = float(formatted_output[0]) > 0.5
        backward = float(formatted_output[1]) > 0.5
        left = float(formatted_output[2]) > 0.5
        right = float(formatted_output[3]) > 0.5

        car_speed = message.car_speed

        if not forward and not backward and abs(car_speed) < 0.3:
            if car_speed < 0:
                backward = True
            else:
                forward = True

        return [
            ("forward", forward),
            ("back", backward),
            ("left", left),
            ("right", right)
        ]

    def process_message(self, message, data_collector):

        car_position = message.car_position

        if data_collector.gate.is_car_through((car_position[0], car_position[2])) and len(data_collector.recorded_data) > 2:
            # data_collector.saveRecord(close_after_save=True)
            data_collector.network_interface.disconnect()

            # Delete the first 3 records, because they are not useful
            del data_collector.recorded_data[:3]
            for snapshot in data_collector.recorded_data:
                print("[", snapshot.current_controls[0], ",", snapshot.current_controls[1], ",", snapshot.current_controls[2], ",", snapshot.current_controls[3], "],")

            QtWidgets.QApplication.quit()
        else:
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionEvaluatePilot(nn_brain.process_message, initial_position=[130,0,-38], initial_angle=-360, initial_speed=30, record=True, record_image=False)
    data_window.gate.set_gate([0, -15], [0, 15], 5)
    app.exec()