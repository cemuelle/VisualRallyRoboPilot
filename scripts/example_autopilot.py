

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


class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True

    def nn_infer(self, message):
        #   Do smart NN inference here


        return [("forward", True)]

    def process_message(self, message, data_collector):

        car_position = message.car_position

        if not data_collector.gate.is_car_through((car_position[0], car_position[2])):
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)
        else:
            # data_collector.saveRecord(close_after_save=True)
            data_collector.onCarControlled("forward", False)
            data_collector.onCarControlled("back", False)
            data_collector.onCarControlled("right", False)
            data_collector.onCarControlled("left", False)
            data_collector.network_interface.disconnect()

            for snapshot in data_collector.recorded_data:
                print("Controls:", snapshot.current_controls)

            QtWidgets.QApplication.quit()

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionEvaluatePilot(nn_brain.process_message, initial_position=[10,0,1], initial_angle=-90, initial_speed=50, record=True, record_image=False)
    data_window.gate.set_gate([-140, -21], [-165, -24], 5)

    app.exec()