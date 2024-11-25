import os.path

from rallyrobopilot import *

from PyQt6.QtCore import Qt, QTimer, QCoreApplication
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6 import uic

from ga.gate import Gate

import pickle
import lzma

class DataCollectionEvaluatePilot(QtWidgets.QMainWindow):
    def __init__(self, message_processing_callback = None, initial_position = [0,0,0], initial_angle = 0, initial_speed = 0, record = True, record_image = False):
        super().__init__()
        self.command_directions = { "w":"forward", "s":"back", "d":"right", "a":"left" }

        self.message_processing_callback = message_processing_callback

        self.network_interface = NetworkDataCmdInterface(self.collectMsg)

        self.timer = QTimer()
        self.timer.timeout.connect(self.network_interface.recv_msg)
        self.timer.start(25)

        self.saving_worker = None

        self.recording = record
        self.record_image = record_image

        self.gate = Gate()

        self.setInitialPosition(initial_position, initial_angle, initial_speed)

        self.recorded_data = []
    def collectMsg(self, msg):
        if self.recording:
            if not self.record_image:
                msg.image = None

            self.recorded_data.append(msg)

        if self.message_processing_callback is not None:
            self.message_processing_callback(msg, self)

    def setInitialPosition(self, position, angle, speed):
        self.network_interface.send_cmd("set position "+ str(position)[1:-1].replace(" ","")+";")
        self.network_interface.send_cmd("set rotation "+ str(angle)+";")
        self.network_interface.send_cmd("set speed "+ str(speed)+";")
        self.network_interface.send_cmd("reset;")


    # def resetNForget(self):

    #     if len(self.recorded_data) == 0:
    #         return

    #     nbr_snapshots_to_forget = self.forgetSnapshotNumber.value() if len(self.recorded_data) > self.forgetSnapshotNumber.value() else len(self.recorded_data)-1

    #     self.recorded_data = self.recorded_data[:-nbr_snapshots_to_forget]

    #     self.network_interface.send_cmd("set position "+ str(self.recorded_data[-1].car_position)[1:-1].replace(" ","")+";")
    #     self.network_interface.send_cmd("set rotation "+ str(self.recorded_data[-1].car_angle)+";")
    #     self.network_interface.send_cmd("reset;")

    #     self.toggleRecord()

    def toggleRecord(self):
        self.recording = not self.recording

    def onCarControlled(self, direction, start):
        command_types = ["release", "push"]
        self.network_interface.send_cmd(command_types[start] + " " + direction+";")

    def saveRecord(self, close_after_save=False):
        if self.saving_worker is not None:
            print("[X] Already saving !")
            return

        if len(self.recorded_data) == 0:
            print("[X] No data to save !")
            return

        if self.recording:
            self.toggleRecord()

        record_name = "record_%d.npz"
        fid = 0
        while os.path.exists(record_name % fid):
            fid += 1

        class ThreadedSaver(QtCore.QThread):
            def __init__(self, path, data):
                super().__init__()
                self.path = path
                self.data = data

            def run(self):
                with lzma.open(self.path, "wb") as f:
                    pickle.dump(self.data, f)

        self.saving_worker = ThreadedSaver(record_name % fid, self.recorded_data)
        self.recorded_data = []
        self.saving_worker.finished.connect(self.onRecordSaveDone)
        if close_after_save:
            self.saving_worker.finished.connect(QCoreApplication.quit)
        self.saving_worker.start()

    def onRecordSaveDone(self):
        print("[+] Recorded data saved to", self.saving_worker.path)
        self.saving_worker = None
