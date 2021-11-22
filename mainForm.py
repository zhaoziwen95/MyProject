# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainForm.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1867, 1000)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_left = QtWidgets.QPushButton(self.centralwidget)
        self.btn_left.setGeometry(QtCore.QRect(0, 930, 471, 90))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(20)
        self.btn_left.setFont(font)
        self.btn_left.setObjectName("btn_left")
        self.btn_right = QtWidgets.QPushButton(self.centralwidget)
        self.btn_right.setGeometry(QtCore.QRect(470, 930, 471, 90))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(20)
        self.btn_right.setFont(font)
        self.btn_right.setObjectName("btn_right")
        self.btn_up = QtWidgets.QPushButton(self.centralwidget)
        self.btn_up.setGeometry(QtCore.QRect(940, 930, 481, 90))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(20)
        self.btn_up.setFont(font)
        self.btn_up.setObjectName("btn_up")
        self.bnt_down = QtWidgets.QPushButton(self.centralwidget)
        self.bnt_down.setGeometry(QtCore.QRect(1420, 930, 451, 90))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(20)
        self.bnt_down.setFont(font)
        self.bnt_down.setObjectName("bnt_down")
        self.bnt_file = QtWidgets.QPushButton(self.centralwidget)
        self.bnt_file.setGeometry(QtCore.QRect(0, 0, 1871, 71))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(36)
        self.bnt_file.setFont(font)
        self.bnt_file.setObjectName("bnt_file")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(0, 70, 1871, 841))
        self.graphicsView.setObjectName("graphicsView")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.btn_right.clicked.connect(MainWindow.right)
        self.bnt_file.clicked.connect(MainWindow.openFile)
        self.btn_left.clicked.connect(MainWindow.left)
        self.btn_up.clicked.connect(MainWindow.up)
        self.bnt_down.clicked.connect(MainWindow.down)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Arbeit Viewer"))
        self.btn_left.setText(_translate("MainWindow", "Left"))
        self.btn_right.setText(_translate("MainWindow", "right"))
        self.btn_up.setText(_translate("MainWindow", "signal"))
        self.bnt_down.setText(_translate("MainWindow", "image"))
        self.bnt_file.setText(_translate("MainWindow", "open file"))
