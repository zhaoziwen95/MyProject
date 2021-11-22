from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, \
    QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import cv2



class Ui_Photo_view(object):
    def setupUi(self, Photo_view):
        Photo_view.setObjectName("Photo_view")
        Photo_view.resize(890, 682)
        self.centralwidget = QtWidgets.QWidget(Photo_view)
        self.centralwidget.setObjectName("centralwidget")
        self.select_button = QtWidgets.QPushButton(self.centralwidget)
        self.select_button.setGeometry(QtCore.QRect(770, 140, 75, 23))
        self.select_button.setCheckable(True)
        self.select_button.setObjectName("select_button")
        self.pic_show = QtWidgets.QGraphicsView(self.centralwidget)
        self.pic_show.setGeometry(QtCore.QRect(0, 0, 711, 641))
        self.pic_show.setObjectName("pic_show")
        Photo_view.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Photo_view)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 890, 23))
        self.menubar.setObjectName("menubar")
        Photo_view.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Photo_view)
        self.statusbar.setObjectName("statusbar")
        Photo_view.setStatusBar(self.statusbar)

        self.retranslateUi(Photo_view)
        self.select_button.clicked.connect(Photo_view.select_button_clicked)
        QtCore.QMetaObject.connectSlotsByName(Photo_view)

    def retranslateUi(self, Photo_view):
        _translate = QtCore.QCoreApplication.translate
        Photo_view.setWindowTitle(_translate("Photo_view", "MainWindow"))
        self.select_button.setText(_translate("Photo_view", "选择文件"))


class show_photo(QMainWindow):
    def __init__(self):
        super(show_photo, self).__init__()
        self.ui = Ui_Photo_view()
        self.ui.setupUi(self)
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 0, 710, 650)

    def select_button_clicked(self):
        file_name = QFileDialog.getOpenFileName(self, "Open File", "./", "png (*.png)")
        image_path = file_name[0]
        if (file_name[0] == ""):
            QMessageBox.information(self, "提示", self.tr("没有选择图片文件！"))
        print(image_path)
        img = cv2.imread(image_path)  # 读取图像
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
    app = QApplication(sys.argv)
    mainWindow = show_photo()
    mainWindow.show()
    sys.exit(app.exec_())
