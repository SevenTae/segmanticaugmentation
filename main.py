#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : main
@Author  : qiqq
@create_time    : 2023/3/31 13:31
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './ui\vesion1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.




from PyQt5.QtWidgets import QFileDialog,QApplication, QMessageBox
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from PyQt5 import QtCore, QtGui, QtWidgets
from core.json2imgcore import json2img
from core.augui import imgaug



from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("SegmentationAug")
        Dialog.resize(945, 749)
        self.imgaug = QtWidgets.QGroupBox(Dialog)
        self.imgaug.setGeometry(QtCore.QRect(20, 20, 881, 421))
        self.imgaug.setObjectName("imgaug")
        self.augstart = QtWidgets.QPushButton(self.imgaug)
        self.augstart.setGeometry(QtCore.QRect(30, 380, 93, 28))
        self.augstart.setObjectName("augstart")
        self.augsavetext = QtWidgets.QTextBrowser(self.imgaug)
        self.augsavetext.setGeometry(QtCore.QRect(110, 330, 621, 31))
        self.augsavetext.setObjectName("augsavetext")
        self.augreset = QtWidgets.QPushButton(self.imgaug)
        self.augreset.setGeometry(QtCore.QRect(190, 380, 93, 28))
        self.augreset.setObjectName("augreset")
        self.augsavebt = QtWidgets.QPushButton(self.imgaug)
        self.augsavebt.setGeometry(QtCore.QRect(10, 330, 93, 28))
        self.augsavebt.setObjectName("augsavebt")
        self.auglabeltext = QtWidgets.QTextBrowser(self.imgaug)
        self.auglabeltext.setGeometry(QtCore.QRect(110, 290, 621, 31))
        self.auglabeltext.setObjectName("auglabeltext")
        self.auglabelbt = QtWidgets.QPushButton(self.imgaug)
        self.auglabelbt.setGeometry(QtCore.QRect(10, 290, 93, 28))
        self.auglabelbt.setObjectName("auglabelbt")
        self.augimgtext = QtWidgets.QTextBrowser(self.imgaug)
        self.augimgtext.setGeometry(QtCore.QRect(110, 250, 621, 31))
        self.augimgtext.setObjectName("augimgtext")
        self.augimgbt = QtWidgets.QPushButton(self.imgaug)
        self.augimgbt.setGeometry(QtCore.QRect(10, 250, 93, 28))
        self.augimgbt.setObjectName("augimgbt")
        self.augmethond = QtWidgets.QGroupBox(self.imgaug)
        self.augmethond.setGeometry(QtCore.QRect(20, 30, 821, 191))
        self.augmethond.setObjectName("augmethond")
        self.gnoise = QtWidgets.QCheckBox(self.augmethond)
        self.gnoise.setGeometry(QtCore.QRect(20, 40, 91, 21))
        self.gnoise.setObjectName("gnoise")
        self.hflip = QtWidgets.QCheckBox(self.augmethond)
        self.hflip.setGeometry(QtCore.QRect(120, 40, 91, 21))
        self.hflip.setObjectName("hflip")
        self.vflip = QtWidgets.QCheckBox(self.augmethond)
        self.vflip.setGeometry(QtCore.QRect(210, 40, 91, 21))
        self.vflip.setObjectName("vflip")
        self.salt = QtWidgets.QCheckBox(self.augmethond)
        self.salt.setGeometry(QtCore.QRect(300, 40, 111, 21))
        self.salt.setObjectName("salt")
        self.cutout = QtWidgets.QCheckBox(self.augmethond)
        self.cutout.setGeometry(QtCore.QRect(440, 40, 111, 21))
        self.cutout.setObjectName("cutout")
        self.contrast = QtWidgets.QCheckBox(self.augmethond)
        self.contrast.setGeometry(QtCore.QRect(540, 40, 111, 21))
        self.contrast.setObjectName("contrast")
        self.bright = QtWidgets.QCheckBox(self.augmethond)
        self.bright.setGeometry(QtCore.QRect(660, 40, 111, 21))
        self.bright.setObjectName("bright")
        self.hue = QtWidgets.QCheckBox(self.augmethond)
        self.hue.setGeometry(QtCore.QRect(20, 80, 111, 21))
        self.hue.setObjectName("hue")
        self.randomrotation = QtWidgets.QCheckBox(self.augmethond)
        self.randomrotation.setGeometry(QtCore.QRect(120, 80, 141, 21))
        self.randomrotation.setObjectName("randomrotation")
        self.shift = QtWidgets.QCheckBox(self.augmethond)
        self.shift.setGeometry(QtCore.QRect(300, 80, 91, 19))
        self.shift.setObjectName("shift")
        self.gblur = QtWidgets.QCheckBox(self.augmethond)
        self.gblur.setGeometry(QtCore.QRect(400, 80, 91, 19))
        self.gblur.setObjectName("gblur")
        self.satura = QtWidgets.QCheckBox(self.augmethond)
        self.satura.setGeometry(QtCore.QRect(500, 80, 111, 21))
        self.satura.setObjectName("satura")
        self.note3 = QtWidgets.QLabel(self.imgaug)
        self.note3.setGeometry(QtCore.QRect(320, 390, 171, 16))
        self.note3.setObjectName("note3")
        self.saveindex = QtWidgets.QLineEdit(self.imgaug)
        self.saveindex.setGeometry(QtCore.QRect(490, 390, 113, 21))
        self.saveindex.setText("")
        self.saveindex.setObjectName("saveindex")
        self.note4 = QtWidgets.QLabel(self.imgaug)
        self.note4.setGeometry(QtCore.QRect(620, 390, 131, 21))
        self.note4.setObjectName("note4")
        self.pernumber = QtWidgets.QLineEdit(self.imgaug)
        self.pernumber.setGeometry(QtCore.QRect(740, 390, 91, 21))
        self.pernumber.setObjectName("pernumber")
        self.json2img = QtWidgets.QGroupBox(Dialog)
        self.json2img.setGeometry(QtCore.QRect(10, 460, 891, 201))
        self.json2img.setObjectName("json2img")
        self.jsondirtext = QtWidgets.QTextBrowser(self.json2img)
        self.jsondirtext.setGeometry(QtCore.QRect(110, 70, 621, 31))
        self.jsondirtext.setObjectName("jsondirtext")
        self.j2isavetext = QtWidgets.QTextBrowser(self.json2img)
        self.j2isavetext.setGeometry(QtCore.QRect(110, 110, 621, 31))
        self.j2isavetext.setObjectName("j2isavetext")
        self.jsondir = QtWidgets.QPushButton(self.json2img)
        self.jsondir.setGeometry(QtCore.QRect(10, 70, 93, 28))
        self.jsondir.setObjectName("jsondir")
        self.json2imgsavebt = QtWidgets.QPushButton(self.json2img)
        self.json2imgsavebt.setGeometry(QtCore.QRect(10, 110, 93, 28))
        self.json2imgsavebt.setObjectName("json2imgsavebt")
        self.j2istart = QtWidgets.QPushButton(self.json2img)
        self.j2istart.setGeometry(QtCore.QRect(30, 170, 93, 28))
        self.j2istart.setObjectName("j2istart")
        self.j2ireset = QtWidgets.QPushButton(self.json2img)
        self.j2ireset.setGeometry(QtCore.QRect(190, 170, 93, 28))
        self.j2ireset.setObjectName("j2ireset")
        self.classbt = QtWidgets.QPushButton(self.json2img)
        self.classbt.setGeometry(QtCore.QRect(10, 30, 93, 28))
        self.classbt.setObjectName("classbt")
        self.classtext = QtWidgets.QTextBrowser(self.json2img)
        self.classtext.setGeometry(QtCore.QRect(110, 30, 621, 31))
        self.classtext.setObjectName("classtext")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "语义分割数据增强"))
        self.imgaug.setTitle(_translate("Dialog", "ImageAug"))
        self.augstart.setText(_translate("Dialog", "开始"))
        self.augreset.setText(_translate("Dialog", "重置"))
        self.augsavebt.setText(_translate("Dialog", "保存路径"))
        self.auglabelbt.setText(_translate("Dialog", "标签文件夹"))
        self.augimgbt.setText(_translate("Dialog", "原图文件夹"))
        self.augmethond.setTitle(_translate("Dialog", "AugMethond"))
        self.gnoise.setText(_translate("Dialog", "G-Niose"))
        self.hflip.setText(_translate("Dialog", "H-Filp"))
        self.vflip.setText(_translate("Dialog", "V-Filp"))
        self.salt.setText(_translate("Dialog", "Salt-Noise"))
        self.cutout.setText(_translate("Dialog", "CutOut"))
        self.contrast.setText(_translate("Dialog", "Contrast"))
        self.bright.setText(_translate("Dialog", "Bright"))
        self.hue.setText(_translate("Dialog", "HUE"))
        self.randomrotation.setText(_translate("Dialog", "Rotation"))
        self.shift.setText(_translate("Dialog", "Shift"))
        self.gblur.setText(_translate("Dialog", "G-blur"))
        self.satura.setText(_translate("Dialog", "Saturation"))
        self.note3.setText(_translate("Dialog", "保存编号(默认从0开始):"))
        self.note4.setText(_translate("Dialog", "每张图像增强为："))
        self.json2img.setTitle(_translate("Dialog", "Json2LabelImg"))
        self.jsondir.setText(_translate("Dialog", "json文件夹"))
        self.json2imgsavebt.setText(_translate("Dialog", "保存路径"))
        self.j2istart.setText(_translate("Dialog", "开始"))
        self.j2ireset.setText(_translate("Dialog", "重置"))
        self.classbt.setText(_translate("Dialog", "类别文件"))

        # 按钮功能区域
        self.augimgbt.clicked.connect(self.augimgbtf)
        self.auglabelbt.clicked.connect(self.auglabelbtf)
        self.augsavebt.clicked.connect(self.augsavebtf)

        self.augreset.clicked.connect(self.augresetf)
        self.j2ireset.clicked.connect(self.j2iresetf)

        self.jsondir.clicked.connect(self.jsondirf)
        self.json2imgsavebt.clicked.connect(self.json2imgsavebtf)

        self.classbt.clicked.connect(self.classbtf)

        self.j2istart.clicked.connect(self.j2istartf)

        self.augstart.clicked.connect(self.augstarttf)

        # 设置编号输入的范围
        int_validator = QIntValidator()
        int_validator.setRange(0, 10000)
        self.saveindex.setValidator(int_validator)

        self.pernumber.setValidator(int_validator)  #每张图被增强成几张

        '''可以选择一张图被增强多少张


        '''

    '''两个reset'''

    def augresetf(self):
        _translate = QtCore.QCoreApplication.translate  # 咱也不知道啥意反正得有
        # 图像增强区域的重置
        self.augimgtext.setText(_translate("Dialog", ""))
        self.auglabeltext.setText(_translate("Dialog", ""))
        self.augsavetext.setText(_translate("Dialog", ""))

    def j2iresetf(self):
        _translate = QtCore.QCoreApplication.translate  # 咱也不知道啥意反正得有
        # json转img的重置
        self.jsondirtext.setText(_translate("Dialog", ""))
        self.j2isavetext.setText(_translate("Dialog", ""))
        self.classtext.setText(_translate("Dialog", ""))

    '''选择文件夹的按钮'''

    # aug的
    def augimgbtf(self):
        _translate = QtCore.QCoreApplication.translate
        imgdir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择原图文件夹路径",
                                                            r"D:\IMPORTANT DATA\DESKTOP\Image\Labelme\JPGImages")
        self.augimgtext.setText(_translate("Dialog", imgdir))

    def auglabelbtf(self):
        _translate = QtCore.QCoreApplication.translate
        labeldir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择标签文件夹路径",
                                                              r"D:\IMPORTANT DATA\DESKTOP\Image\Labelme\JPGImages")
        self.auglabeltext.setText(_translate("Dialog", labeldir))

    def augsavebtf(self):
        _translate = QtCore.QCoreApplication.translate
        augsavedir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择保存路径",
                                                                r"D:\IMPORTANT DATA\DESKTOP\Image\Labelme\JPGImages")
        self.augsavetext.setText(_translate("Dialog", augsavedir))

    # j2i的
    def jsondirf(self):
        _translate = QtCore.QCoreApplication.translate
        jsondir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择文件夹",
                                                             r"D:\IMPORTANT DATA\DESKTOP\Image\Labelme\JPGImages")
        self.jsondirtext.setText(_translate("Dialog", jsondir))

    def json2imgsavebtf(self):
        _translate = QtCore.QCoreApplication.translate
        savedir = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择保存路径",
                                                             r"D:\IMPORTANT DATA\DESKTOP\Image\Labelme\JPGImages")
        self.j2isavetext.setText(_translate("Dialog", savedir))

    '''类别文件'''

    def classbtf(self):
        _translate = QtCore.QCoreApplication.translate
        classfiles = QtWidgets.QFileDialog.getOpenFileName(None, "请选择类别文件",
                                                           r"D:\IMPORTANT DATA\DESKTOP\Image\Labelme\JPGImages")
        file = classfiles[0]
        self.classtext.setText(_translate("Dialog", file))

    '''两个开始按钮'''

    def j2istartf(self):
        # print("你好啊")
        #

        clsfile = self.classtext.toPlainText()
        jsonpath = self.jsondirtext.toPlainText()
        savepath = self.j2isavetext.toPlainText()  # 注意这里保存的时候要自己建一个空白文件夹
        if clsfile == "" or jsonpath == "" or savepath == "":
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '请检查JSon2LableImg中的路径是否存在空')
            msg_box.exec_()
        else:

            json2img(clsfile, json_dir=jsonpath, savedir=savepath)
            msg_box = QMessageBox(QMessageBox.Information, '提示', 'json转换为标签图完成')
            msg_box.exec_()

    # aug
    def augstarttf(self):

        # 文件夹
        sourceimg = self.augimgtext.toPlainText()
        sourelabel = self.auglabeltext.toPlainText()
        savedir = self.augsavetext.toPlainText()
        saveindexx = self.saveindex.text()  # 这个索引并只是在保存的时候当第几次进行增强的索引
        pernumber = self.pernumber.text()  # 这个索引并只是在保存的时候当第几次进行增强的索引
        # 增强方式
        augchoise = []
        if saveindexx == "":
            saveindexx = "0"

        if pernumber == "":
            pernumber = "0"

        # print("index",saveindexx)
        if self.gnoise.isChecked():
            augchoise.append("gnoise")
        if self.hflip.isChecked():
            augchoise.append("hflip")
        if self.vflip.isChecked():
            augchoise.append("vflip")
        if self.salt.isChecked():
            augchoise.append("saltnoise")
        if self.cutout.isChecked():
            augchoise.append("cutout")
        if self.contrast.isChecked():
            augchoise.append("contrast")
        if self.bright.isChecked():
            augchoise.append("bright")
        if self.hue.isChecked():
            augchoise.append("hue")
        if self.randomrotation.isChecked():
            augchoise.append("Rotation")
        if self.shift.isChecked():
            augchoise.append("shift")
        if self.gblur.isChecked():
            augchoise.append("gblur")
        if self.satura.isChecked():
            augchoise.append("satura")


        if sourceimg == "" or sourelabel == "" or savedir == "":
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '请检查Aug原图、标签、保存路径是否正确')
            msg_box.exec_()
        elif len(augchoise) == 0:
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '未选择数据增强方式')
            msg_box.exec_()
        else:
            #
            imgaug(sourceimg, sourelabel, savedir, augchoise, saveindexx,pernumber=pernumber)
            msg_box = QMessageBox(QMessageBox.Information, '提示', '数据增强完成')
            msg_box.exec_()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())