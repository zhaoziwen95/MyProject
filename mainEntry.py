import sys
import cv2

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage

from PIL import Image

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from mainForm import Ui_MainWindow
import os
from processImg import processImg
from FigureCanvas import Figure_Canvas
import matplotlib
from matplotlib import pylab as plt
from main_dev.PYTHON.general.tools.import_measurement_data_matlab import import_signal_data_matlab
import numpy as np
from main_dev.PYTHON.general.signals.time_signal import TimeSignal
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtWidgets import QMainWindow,QApplication
import sys

class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 70, 1871, 841)
        self.flag=0

    def filePointer(self):
        self.ls = os.listdir(self.datasetAddress)
        self.ls.sort()
        self.fileP = self.ls.index(self.filenameLocal)

    def left(self):
        if self.flag == 0:
            filename = self.ls[self.fileP - 5]
            self.bnt_file.setText(filename)
            self.fileP = self.fileP - 5
            img = cv2.imread(self.datasetAddress + '/' + filename)  # 读取图像
            img = processImg(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            self.zoomscale = 1  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.view.setScene(self.scene)
            self.view.show()
        else:
            dr = Figure_Canvas()
            # 实例化一个FigureCanvas
            dr.test(self.datasetAddress,self.ls[self.fileP -5])  # 画图
            self.scene = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
            self.scene.addWidget(dr)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
            self.view.setScene(self.scene)  # 第五步，把QGraphicsScene放入QGraphicsView
            self.view.show()  # 最后，调用show方法呈现图形！Voila!!
            # self.setCentralWidget(self.view)
            # self.view.setFixedSize(1600, 600)
            self.flag = 1

    def right(self):
        if self.flag==0:
            filename = self.ls[self.fileP + 5]
            self.bnt_file.setText(filename)
            self.fileP = self.fileP + 5
            img = cv2.imread(self.datasetAddress + '/' + filename)  # 读取图像
            img = processImg(img)
            # self.filePointer()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            self.zoomscale = 1  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.view.setScene(self.scene)
            self.view.show()
        else:
            dr = Figure_Canvas()
            # 实例化一个FigureCanvas
            dr.test(self.datasetAddress,self.ls[self.fileP +5])  # 画图
            self.scene = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
            self.scene.addWidget(dr)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
            self.view.setScene(self.scene)  # 第五步，把QGraphicsScene放入QGraphicsView
            self.view.show()  # 最后，调用show方法呈现图形！Voila!!
            # self.setCentralWidget(self.view)
            # self.view.setFixedSize(1600, 600)
            self.flag = 1


    def up(self):
        if self.flag==0:
            dr = Figure_Canvas()
            # 实例化一个FigureCanvas
            dr.test(self.datasetAddress,self.ls[self.fileP -1])  # 画图
            self.scene = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
            self.scene.addWidget(dr)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
            self.view.setScene(self.scene)  # 第五步，把QGraphicsScene放入QGraphicsView
            self.view.show()  # 最后，调用show方法呈现图形！Voila!!
            # self.setCentralWidget(self.view)
            # self.view.setFixedSize(1600, 600)
            self.flag = 1
    def down(self):
        if self.flag==1:
            filename = self.ls[self.fileP]
            self.bnt_file.setText(filename)
            self.fileP = self.fileP
            self.flag=0
            img = cv2.imread(self.datasetAddress + '/' + filename)  # 读取图像
            img = processImg(img)
            # self.filePointer()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            self.zoomscale = 1  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.view.setScene(self.scene)
            self.view.show()
            self.flag=0

    def openFile(self):

        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "./", "png (*.png)")# open a file filename is absolute address
        self.bnt_file.setText(filename) #show filename on butten
        self.datasetAddress = filename.rpartition("/")[0] # splipt filename in two parts ,dataset address and local address
        self.filenameLocal = filename.rpartition("/")[2]
        self.filePointer() # it is like a pointer, which point the file you visit, self.fileP is parameter
        filename = self.ls[self.fileP]  # filepointer points the file you choosed. the address of next image
                                        # is ls[self.fileP+5]
        if self.flag==0:    # if the current image is picture, not the signal

            img = cv2.imread(self.datasetAddress + '/' + filename)
            img = processImg(img) #input a original image, return a image, which have original image and processd one
            # self.filePointer()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
            x = img.shape[1]  # 获取图像大小
            y = img.shape[0]
            self.zoomscale = 1  # 图片放缩尺度
            frame = QImage(img, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)
            self.view.setScene(self.scene)
            self.view.show()






if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()

    window.show()

    sys.exit(app.exec_())
