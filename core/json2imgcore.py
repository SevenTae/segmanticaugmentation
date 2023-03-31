#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : json2imgcore
@Author  : qiqq
@create_time    : 2023/3/31 13:35
"""


import os
import json
import os.path as osp
from labelme import utils
import  numpy as  np
def json2img(cla,json_dir,savedir):
    '''

    :param cla:
    :param jpgdir: 原图文件夹 好像没有也行
    :param json_dir:  原图对应的json
    :param savedir: json转成单通道标签的保存路径
    progressBar:用来控制qt里边的进度条
    :return:
    '''

    '''cla是类别文件必须是txt形式的且里边的内容一行就是一个类别，其中背景类在第一行记作_background_'''
    classes =[]
    with open(cla,mode='r',encoding='utf-8') as f:
        for line in f.readlines():
            classes.append(line.strip("\n"))

    f.close()

    #

    count = os.listdir(json_dir)

    realjson_file =[]
    for i in range(0, len(count)):

        path = os.path.join(json_dir, count[i])

        if os.path.isfile(path) and path.endswith('.json'):
            realjson_file.append(path)


    # #
    for i in range(0, len(realjson_file)):

        path = os.path.join(json_dir, realjson_file[i])

        data = json.load(open(path))

        # if data['imageData']: #这个玩意完整的编码了整个原始图像的信息
        #     imageData = data['imageData']
        # else:
        #     imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
        #     with open(imagePath, 'rb') as f:
        #         imageData = f.read()
        #         imageData = base64.b64encode(imageData).decode('utf-8')

        # img = utils.img_b64_to_arr(imageData)
        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        imageHeight =data.get("imageHeight")
        imageWidth =data.get("imageWidth")
        img_shape= (imageHeight,imageWidth,3)



        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))

        lbl = utils.shapes_to_label(img_shape, data['shapes'], label_name_to_value)

        # PIL.Image.fromarray(img).save(osp.join(jpgdir, count[i].split(".")[0] + '.jpg'))

        new = np.zeros([imageHeight, imageWidth])
        for name in label_names:
            index_json = label_names.index(name)
            index_all = classes.index(name)
            new = new + index_all * (np.array(lbl) == index_json)

        utils.lblsave(osp.join(savedir, count[i].split(".")[0] + '.png'), new)
        print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')




