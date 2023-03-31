#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : augui
@Author  : qiqq
@create_time    : 2023/3/30 16:35
"""


import os
import  numpy as np
import os
from pathlib import Path
import  matplotlib.pyplot as  plt
from PIL import Image
from .augcore   import SegmantationAugChoice,decode_segmap
from tqdm import tqdm
from collections import namedtuple

#
# def get_putpalette(Clss, color_other=[0, 0, 0]):
#         '''
#         灰度图转8bit彩色图
#         :param Clss:颜色映射表
#         :param color_other:其余颜色设置
#         :return:
#         '''
#         putpalette = []
#         for cls in Clss:
#             putpalette += list(cls.color)
#         putpalette += color_other * (255 - len(Clss))
#         return putpalette
#
# Cls = namedtuple('cls', ['name', 'id', 'color'])
# Clss = [
#     Cls('backgroud', 0, (0, 0, 0)),
#     Cls('pv', 1, (0, 255, 0)),
#     Cls('ignore', 255, (255, 255, 255))
# ]
# bin_colormap = get_putpalette(Clss)


def imgaug(sourceimg,sourcelabl,savedir,augchoise,saveindex=0,pernumber=5):
    '''
    原图和标签的名字必须一样（除了前缀）
    :param sourceimg: 原img文件夹
    :param sourcelabl: 原label文件夹
    :param savedir:保存的文件夹
    :param augchoise:选择要增强的方式（多选就是每一张图都混合增强）
    :param saveindex:保存的索引默认是0
    :param pernumber:一张要增强为几张
    :return:
    '''
    augdict = {
        "hflip": "HorizontalFlip",
        "vflip": "VerticalFlip",
        "Rotation": "RandomRotation",
        "shift": "RandomShift",
        "gblur": "RandomGaussian",
        "saltnoise": "RandomSaltNoise",
        "bright": "RandomBright",
        "contrast": "RandomContrast",
        "satura": "RandomSaturation",
        "hue": "RandomHUE",
        "cutout": "RandomCutout",
        "gnoise": "RandomGaussianNoise",

    }


    palette = [0, 0, 0, 0, 255, 0,0,255,255]

    ##先在savedir里边建立两个子文件夹用来增强后的文件 augimgdir，auglabeldir
    augimgdir= os.path.join(savedir,"augimgdir")
    auglabeldir= os.path.join(savedir,"auglabeldir")
    Path(augimgdir).mkdir(parents=True, exist_ok=True)
    Path(auglabeldir).mkdir(parents=True, exist_ok=True)


    sourceimglist = [i for i in os.listdir(sourceimg) if i.endswith(".jpg") or i.endswith(".jpeg") or i.endswith(".bmp") or i.endswith(".tif") ]
    sourcelabellist = [i for i in os.listdir(sourcelabl) if i.endswith(".png") or i.endswith(".tif") or i.endswith(".bmp")]
    print("可用的一共",len(sourceimglist))
    #注意这里暂时设置未原图和标签一样的前缀名字
    for index,i in enumerate(sourceimglist):
        image_name= i.split(".")[0]
        image_type=i.split(".")[1]
        label_type=sourcelabellist[index].split(".")[1]
        img=Image.open(os.path.join(sourceimg,i))
        lab=Image.open(os.path.join(sourcelabl,image_name+"."+label_type))

        image = np.array(img)
        label = np.array(lab)
        count = saveindex
        for j in range(pernumber):
            augimg=image
            auglabel=label
            for k in augchoise:
                augimg,auglabel =eval('SegmantationAugChoice.'+augdict[k])(augimg,auglabel)  #出来的还是numpy，每循环一次都会叠加一次
            #保存

            # laba=decode_segmap(auglabel, dataset='pascal_customer')
            #
            # plt.imshow(augimg)
            # plt.show()
            # plt.imshow(laba)
            # plt.show()
            #
            imgsave = Image.fromarray(np.uint8(augimg))
            imgsave.save(os.path.join(augimgdir , image_name +"aug_"+ str(count)+ '.'+image_type))

            labelsave=Image.fromarray(np.uint8(auglabel)).convert('P')

            labelsave.putpalette(palette)
            labelsave.save(os.path.join(auglabeldir ,image_name +"aug_"+ str(count)+'.png'))

            print(f"数据集共:{len(sourceimglist)},第{index+1}张中的,{j+1}/{pernumber}")
            count =count +1



if __name__ == '__main__':

    sourceimg=r"D:\IMPORTANT DATA\DESKTOP\数据增强样例\原图"
    sourcelabl=r"D:\IMPORTANT DATA\DESKTOP\数据增强样例\标签"
    savedir=r"D:\IMPORTANT DATA\DESKTOP\数据增强样例"
    augchoise=["Rotation"]
    imgaug(sourceimg, sourcelabl, savedir, augchoise, saveindex=0, pernumber=2)

