# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 434)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.openFileDialog = QtWidgets.QPushButton(self.centralwidget)
        self.openFileDialog.setGeometry(QtCore.QRect(10, 10, 131, 51))
        self.openFileDialog.setObjectName("openFileDialog")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(150, 10, 641, 411))
        self.graphicsView.setObjectName("graphicsView")
        self.findObject = QtWidgets.QPushButton(self.centralwidget)
        self.findObject.setGeometry(QtCore.QRect(10, 70, 131, 51))
        self.findObject.setObjectName("findObject")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openFileDialog.setText(_translate("MainWindow", "Открыть изображение"))
        self.findObject.setText(_translate("MainWindow", "Найти объекты"))

