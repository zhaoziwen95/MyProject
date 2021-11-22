import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication
import sys
from matplotlib import pylab as plt
from main_dev.PYTHON.general.tools.import_measurement_data_matlab import import_signal_data_matlab
import numpy as np
from main_dev.PYTHON.general.signals.time_signal import TimeSignal


class Figure_Canvas(
    FigureCanvas):  # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，

    # def __init__(self, parent=None, width=11, height=5, dpi=100):
    def __init__(self, path="/home/ziwen/Desktop/data", file="2021_05_21__100834_calib1"):
        fig = Figure(figsize=(25, 10), dpi=100)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(None)
        # self.path=path
        # self.file=file



        self.axes1 = fig.add_subplot(4,4,1)  # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.axes2 = fig.add_subplot(4,4,2)
        self.axes3 = fig.add_subplot(4,4,3)
        self.axes4 = fig.add_subplot(4,4,4)
        self.axes5 = fig.add_subplot(4,4,5)
        self.axes6 = fig.add_subplot(4,4,6)
        self.axes7 = fig.add_subplot(4,4,7)
        self.axes8 = fig.add_subplot(4,4,8)
        self.axes9 = fig.add_subplot(4,4,9)
        self.axes10 = fig.add_subplot(4,4,10)
        self.axes11 = fig.add_subplot(4,4,11)
        self.axes12 = fig.add_subplot(4,4,12)
        self.axes13 = fig.add_subplot(4,4,13)
        self.axes14 = fig.add_subplot(4,4,14)
        self.axes15 = fig.add_subplot(4,4,15)
        self.axes16 = fig.add_subplot(4,4,16)

    def test(self,path,file):
        # path = "/home/ziwen/Desktop/data"
        # file = "2021_05_21__100834_calib1"

        file=file[:-4]
        print(path)
        print(file)
        # Messdatei einlesen als "TimeSignal"-Objekt (numpy-array)
        time_signal = import_signal_data_matlab(path, file)

        # Nicht benötigte Dimensionen weglassen
        time_signal2 = TimeSignal(np.squeeze(np.array(time_signal)), f_samp=time_signal.f_samp)

        # Signal von Sensor 0 (Sender) zu Sensor 2 (Empfänger) plotten
        begin_time = 1000
        end_time = 12000
        self.axes1.plot(time_signal2[0,0,begin_time:end_time])
        self.axes2.plot(time_signal2[0,1,begin_time:end_time])
        self.axes3.plot(time_signal2[0,2,begin_time:end_time])
        self.axes4.plot(time_signal2[0,3,begin_time:end_time])
        self.axes5.plot(time_signal2[1,0,begin_time:end_time])
        self.axes6.plot(time_signal2[1,1,begin_time:end_time])
        self.axes7.plot(time_signal2[1,2,begin_time:end_time])
        self.axes8.plot(time_signal2[1,3,begin_time:end_time])
        self.axes9.plot(time_signal2[2,0,begin_time:end_time])
        self.axes10.plot(time_signal2[2,1,begin_time:end_time])
        self.axes11.plot(time_signal2[2,2,begin_time:end_time])
        self.axes12.plot(time_signal2[2,3,begin_time:end_time])
        self.axes13.plot(time_signal2[3,0,begin_time:end_time])
        self.axes14.plot(time_signal2[3,1,begin_time:end_time])
        self.axes15.plot(time_signal2[3,2,begin_time:end_time])
        self.axes16.plot(time_signal2[3,3,begin_time:end_time])
##########################################################################################################

# from matplotlib import pylab as plt
# from main_dev.PYTHON.general.tools.import_measurement_data_matlab import import_signal_data_matlab
# import numpy as np
# from main_dev.PYTHON.general.signals.time_signal import TimeSignal
#
# # Pfad zu den Messdaten angeben
# path = "/home/ziwen/Desktop/data"
# file = "2021_05_21__100834_calib1"
#
# # Messdatei einlesen als "TimeSignal"-Objekt (numpy-array)
# time_signal = import_signal_data_matlab(path, file)
#
# # Nicht benötigte Dimensionen weglassen
# time_signal2= TimeSignal(np.squeeze(np.array(time_signal)), f_samp=time_signal.f_samp)
#
# # Signal von Sensor 0 (Sender) zu Sensor 2 (Empfänger) plotten
# begin_time=1000
# end_time=12000
# ax1=plt.subplot(4,4,1)
# plt.plot(time_signal2[0,0,begin_time:end_time])
# ax2=plt.subplot(4,4,2)
# plt.plot(time_signal2[0,1,begin_time:end_time])
# ax1=plt.subplot(4,4,3)
# plt.plot(time_signal2[0,2,begin_time:end_time])
# ax2=plt.subplot(4,4,4)
# plt.plot(time_signal2[0,3,begin_time:end_time])
# ax1=plt.subplot(4,4,5)
# plt.plot(time_signal2[1,0,begin_time:end_time])
# ax2=plt.subplot(4,4,6)
# plt.plot(time_signal2[1,1,begin_time:end_time])
# ax1=plt.subplot(4,4,7)
# plt.plot(time_signal2[1,2,begin_time:end_time])
# ax2=plt.subplot(4,4,8)
# plt.plot(time_signal2[1,3,begin_time:end_time])
# ax1=plt.subplot(4,4,9)
# plt.plot(time_signal2[2,0,begin_time:end_time])
# ax2=plt.subplot(4,4,10)
# plt.plot(time_signal2[2,1,begin_time:end_time])
# ax1=plt.subplot(4,4,11)
# plt.plot(time_signal2[2,2,begin_time:end_time])
# ax2=plt.subplot(4,4,12)
# plt.plot(time_signal2[2,3,begin_time:end_time])
# ax1=plt.subplot(4,4,13)
# plt.plot(time_signal2[3,0,begin_time:end_time])
# ax2=plt.subplot(4,4,14)
# plt.plot(time_signal2[3,1,begin_time:end_time])
# ax1=plt.subplot(4,4,15)
# plt.plot(time_signal2[3,2,begin_time:end_time])
# ax2=plt.subplot(4,4,16)
# plt.plot(time_signal2[3,3,begin_time:end_time])
# plt.show()
