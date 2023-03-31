#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : augcore
@Author  : qiqq
@create_time    : 2023/3/30 14:02
"""

from skimage import util, img_as_float, io,transform ,exposure ,img_as_ubyte  # 导入所需要的 skimage 库
import os
import  numpy as np
from  PIL import Image
import cv2 as cv
import  matplotlib.pyplot as  plt

'''

水平翻转b，上下翻转b，高斯模糊b，色彩变换（亮度，对比度，饱和度，色调）
，cutout(先不要了)
仿射变换（平移变换b、旋转变换d、尺度变换(先不要了)、）

'''
#水平翻转

augdict={
    "hflip":"HorizontalFlip",
    "vflip":"VerticalFlip",
    "Rotation":"RandomRotation",
    "shift":"RandomShift",
    "gblur":"RandomGaussian",
    "gnoise":"RandomGaussianNoise",
    "saltnoise":"RandomSaltNoise",
    "bright":"RandomBright",
    "contrast":"RandomContrast",
    "satura":"RandomSaturation",
    "hue":"RandomHUE",
    "cutout":"RandomCutout",

}


# 为了服务后续的颜色工作的
def convert(image, alpha=1, beta=0):
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def rgb2hsv(image):
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)


def hsv2rgb(image):
    return cv.cvtColor(image, cv.COLOR_HSV2RGB)


#######
class SegmantationAugChoice():
    def __init__(self):
        pass

    @classmethod
    def HorizontalFlip(cls,image,label):

            img_hf =cv.flip(image, 1)
            mask_hf =cv.flip(label, 1)

            return img_hf,mask_hf

    #垂直翻转
    @classmethod
    def  VerticalFlip(cls,image,label):

            img_vf = cv.flip(image, 0)
            mask_vf =cv.flip(label, 0)

            return img_vf,mask_vf


    #随机旋转 √
    @classmethod
    def RandomRotation(cls,image,label):

            '''

            :param angle_upper: 旋转角度
            # :param rotation_prob: 旋转概率
            :param img_fill_value: 原图旋转后用什么像素值填充
            :param seg_fill_value: 标签图旋转后用什么像素值填充
            :param img_interpolation:原图的插值方式
            :param seg_interpolation: 标签图的插值方式
            '''
            angle_upper = 60
            img_fill_value = 0.0
            seg_fill_value = 255
            img_interpolation = 'bicubic'
            seg_interpolation = 'nearest'
            # interpolation to cv2 interpolation
            interpolation_dict = {
                'nearest': cv.INTER_NEAREST,
                'bilinear': cv.INTER_LINEAR,
                'bicubic': cv.INTER_CUBIC,
                'area': cv.INTER_AREA,
                'lanczos': cv.INTER_LANCZOS4
            }

            h_ori, w_ori = image.shape[:2]
            rand_angle = np.random.randint(-1*angle_upper, angle_upper)
            matrix = cv.getRotationMatrix2D(center=(w_ori / 2, h_ori / 2), angle=rand_angle, scale=1)
            image_ro = cv.warpAffine(image, matrix, (w_ori, h_ori), flags=interpolation_dict[img_interpolation], borderValue=img_fill_value)
            segmentation_ro = cv.warpAffine(label, matrix, (w_ori, h_ori), flags=interpolation_dict[seg_interpolation], borderValue=seg_fill_value)

            return image_ro,segmentation_ro

    #随机平移
    @classmethod
    def RandomShift(cls,image,label):
        '''

              :param img_fill_value: 原图旋转后用什么像素值填充
              :param seg_fill_value: 标签图旋转后用什么像素值填充
              :param img_interpolation:原图的插值方式
              :param seg_interpolation: 标签图的插值方式
              '''
        tx_max = 100  #平移的最大像素数
        ty_max = 100
        img_fill_value = 0.0
        seg_fill_value = 255
        img_interpolation = 'bicubic'
        seg_interpolation = 'nearest'
        # interpolation to cv2 interpolation
        interpolation_dict = {
            'nearest': cv.INTER_NEAREST,
            'bilinear': cv.INTER_LINEAR,
            'bicubic': cv.INTER_CUBIC,
            'area': cv.INTER_AREA,
            'lanczos': cv.INTER_LANCZOS4
        }

        h_ori, w_ori = image.shape[:2]
        tx = np.random.randint(-1 * tx_max, ty_max)
        ty = np.random.randint(-1 * ty_max, ty_max)
        #像素点 (x,y) 沿 x 轴平移 dx、沿 y 轴平移 dy，公式：
        matrix = np.float32([[1,0,tx],[0,1,ty]])
        image_shi = cv.warpAffine(image, matrix, (w_ori, h_ori), flags=interpolation_dict[img_interpolation],
                                 borderValue=img_fill_value)
        segmentation_shi = cv.warpAffine(label, matrix, (w_ori, h_ori), flags=interpolation_dict[seg_interpolation],
                                        borderValue=seg_fill_value)

        return image_shi, segmentation_shi

    #高斯模糊
    @classmethod
    def RandomGaussian(cls,image,label):

        kenermax=[3,5,7]#最大高斯核
        sigmmaX=[0,0.5,1,1.5,2]
        sigmmaY=[0,0.5,1,1.5,2]
        kernelsize = kenermax[np.random.randint(0, 2)]
        sigmx=sigmmaX[np.random.randint(0, 4)]
        sigmy=sigmmaY[np.random.randint(0, 4)]
        img_g=cv.GaussianBlur(image, (kernelsize,kernelsize), sigmaX=sigmx,sigmaY=sigmy)
        # img_g=util.random_noise(image, mode="gaussian")

        return img_g,label

    #随机椒盐噪声
    @classmethod
    def RandomSaltNoise(cls,image,label):

        rows, cols, channels = image.shape
        # 定义噪声比例（即噪声像素占总像素的比例）
        a=[0.01,0.02,0.05,0.002,0.003,0.1]
        noise_ratio = np.random.choice( a )
        # 添加椒盐噪声
        noise_img = np.copy(image)
        noise_count = int(rows * cols * noise_ratio)
        for i in range(noise_count):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            # 随机选择黑色或白色像素
            if np.random.random() < 0.5:
                noise_img[x, y] = [0, 0, 0]  # 黑色像素
            else:
                noise_img[x, y] = [255, 255, 255]  # 白色像素

        return noise_img,label


    #随机高斯噪声
    @classmethod
    def RandomGaussianNoise(cls,image,label):

        # 定义高斯噪声的均值和标准差
        mean = 0
        # stddev =10
        stddev =[5,10,15,20,25,30]

        # 生成与原图像相同大小的随机高斯噪声图像
        noise = np.random.normal(mean, np.random.choice(stddev), image.shape)

        # 将原图像转换为浮点数类型，并加上高斯噪声
        img_with_noise = image.astype(np.float32) + noise

        # 将图像数据裁剪到[0, 255]范围内
        img_with_noise = np.clip(img_with_noise, 0, 255)

        # 将图像数据转换回整型类型并保存
        img_with_noise = img_with_noise.astype(np.uint8)

        return img_with_noise,label

    #随机cutout
    @classmethod
    def RandomCutout(cls,image,label):
        # 随机生成cutout区域的大小和位置
        image_cut=image
        label_cut=label
        min_size=10
        max_size=100
        h, w, _ = image.shape
        x = np.random.randint(0, w - 1)
        y = np.random.randint(0, h - 1)
        size = np.random.randint(min_size, max_size)

        x1 = max(x - size // 2, 0)
        y1 = max(y - size // 2, 0)
        x2 = min(x + size // 2, w - 1)
        y2 = min(y + size // 2, h - 1)

        # 在原图上进行cutout并用黑色填充
        image_cut[y1:y2, x1:x2, :] = 0

        # 在标签图上进行cutout的地方是背景
        label_cut[y1:y2, x1:x2] = 0

        return image_cut, label_cut


        #
        # # 读取原图和标签图
        #
        #
        # # 定义cutout数据增强器
        # cutout = iaa.Cutout(nb_iterations=1, size=0.2, squared=False,cval=255)
        #
        # # 将原图和标签图合并为一个数组
        # combined = np.concatenate((image, np.expand_dims(label, axis=2)), axis=2)
        #
        # # 对合并后的数组进行cutout数据增强
        # augmented = cutout.augment_image(combined)
        #
        # # 将增强后的数组分离为原图和标签图
        # augmented_img = augmented[:, :, :3]
        # augmented_label = augmented[:, :, 3]
        #
        # # 将标签图转换回灰度图像
        # augmented_label = augmented_label.astype(np.uint8)
        #
        # return augmented_img,augmented_label



    #随机亮度增强
    @classmethod
    def RandomBright(cls,image,label):
        brightness_delta = 50
        beta=np.random.uniform(-1*brightness_delta,brightness_delta)
        img_b=convert(image,beta=beta)
        return img_b,label

    #随机对比度增强
    @classmethod
    def RandomContrast(cls,image,label):
        contrast_range = (0.5, 1.5)
        contrast_lower, contrast_upper = contrast_range
        img_contrast=convert(image, alpha=np.random.uniform(contrast_lower, contrast_upper))

        return img_contrast,label


    #随机饱和度
    @classmethod
    def RandomSaturation(cls,image,label):
        saturation_range=(0.5,2.5)
        saturation_lower, saturation_upper = saturation_range
        image = rgb2hsv(image)
        image[..., 1] = convert(image[..., 1], alpha=np.random.uniform(saturation_lower, saturation_upper))
        image_s = hsv2rgb(image)

        return image_s,label


    #随机色调
    @classmethod
    def RandomHUE(cls,image,label):
        hue_delta = [10,11,12,13,14,15,16,17,18,19,20]
        image = rgb2hsv(image)
        image[..., 0] = (image[..., 0].astype(int) + np.random.randint(-1*np.random.choice(hue_delta), np.random.choice(hue_delta))) % 180
        image_hue = hsv2rgb(image)

        return image_hue,label



#这俩是辅助查看的
def get_pascal_customer_labels():
    """
    '''如果有对应颜色就换成对应颜色，如果没有对应颜色且类别数小于21的话就默认'''
    Returns:
        np.ndarray with dimensions (21, 3)

    """
    # return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    #                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    #                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    #                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    #                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    #                    [0, 64, 128]])
    return np.asarray([[255 ,0, 0], [255 ,255, 0], [192 ,192, 0], [0 ,255, 0],
                       [128,128,128], [0, 0 ,255]])
def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labelss into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial_module location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    if dataset =='pascal_customer':
        n_classes = 6
        label_colours = get_pascal_customer_labels()

    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb




if __name__ == '__main__':
    image_path = r"D:\IMPORTANT DATA\DESKTOP\数据增强样例\原图\1_24.jpg"
    segmap_path = r"D:\IMPORTANT DATA\DESKTOP\数据增强样例\标签\1_24.png"

    # img=Image.open(image_path)
    # label=Image.open(segmap_path)
    # img=np.array(img)
    # label=np.array(label)
    # plt.title("yuanimage")
    # plt.imshow(img)
    # plt.show()
    # plt.title("yuanlabel")
    # plt.imshow(label)
    # plt.show()
    #
    # ai,li=RandomCutout(img,label)
    # segmap = decode_segmap(li, dataset='pascal_customer')
    # plt.title("augimg")
    # plt.imshow(ai)
    # plt.show()
    # plt.title("auglabel")
    # plt.imshow(segmap)
    # plt.show()

