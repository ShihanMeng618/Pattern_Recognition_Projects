#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference: https://www.jianshu.com/p/b9f2c92dfeaa
import numpy as np
import csv
from PIL import Image

def array_to_image(array):
    array = array*255
    new_image = Image.fromarray(array.astype(np.uint8))
    return new_image

def plot_images(images, col, row, each_width, each_height, new_type):
    new_image = Image.new(new_type, (col*each_width, row*each_height))
    for i in range(len(images)):
        each_img = array_to_image(images[i].reshape(each_width, each_height))
        box = ((i % col) * each_width, (i // col)*each_width) # 向下取整，upper left point的横纵坐标
        new_image.paste(each_img, box)
    return new_image


def pca(data_mat, top_n_feat = 1):
    '''
    PCA function
    :param data_mat: input data
    :param top_n_feat: 保留的特征数，题目中要求top1
    :return: 降维后的数据集res_data和原始数据被重构后的矩阵recon
    '''
    # 零均值化
    mean = data_mat - data_mat.mean(axis=0) # 行均值
    # 计算协方差矩阵
    cov = np.cov(mean, rowvar=False) # 784*784
    # 计算特征值
    eig_lambda, eig_vects = np.linalg.eig(cov)
    eig_vects = eig_vects[eig_lambda.argsort()] # 按lambda对vectors排序 784*784
    res_eig_vects = eig_vects[:,783]

    res_data = mean.dot(res_eig_vects) # 1000 * 1(top_n_feat)
    recon = np.dot(res_data.reshape(-1,1), res_eig_vects.T.reshape(1,-1)) + data_mat.mean(axis=0) # 1000*784

    return res_data, recon


def main():
    #import data from MNIST each class 1000 images
    with open('mnist_train.csv', newline='') as csv_file1:
        train_data_lines = csv.reader(csv_file1)
        train_dataset = list(train_data_lines)
        train_matrix = np.array(train_dataset).astype("float32")
        x_train = train_matrix[:,1:785]/255 # normalization
        y_train = train_matrix[:,0].reshape(-1,1)

    images = np.hstack([y_train, x_train]) # 60000*785
    images = images[images[:,0].argsort()] # 按label对行排序
    count = np.zeros(10)
    data = np.zeros([10000,785]) # 10 * 1000*785
    count_sum = 0
    for label in range(10):
        count[label] = np.sum(images[:,0] == label)
        data[label*1000:(label+1)*1000,:] = images[int(count_sum):int(count_sum+1000),:]
        count_sum += count[label]

    # PCA
    for label in range(10):
        origin_images = data[label*1000:(label+1)*1000,1:785]
        res_data, recon = pca(origin_images)
        res_images = plot_images(recon[0:100,:], 10, 10, 28, 28, 'L')
        origin_imgs = plot_images(origin_images[0:100,:], 10, 10, 28, 28, 'L')
        res_images.save(str(label)+".png")
        origin_imgs.save(str(label)+"origin.png")


if __name__ == "__main__":
    main()