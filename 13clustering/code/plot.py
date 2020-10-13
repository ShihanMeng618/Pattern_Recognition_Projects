#!/usr/bin/python
# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'  # 用来正常显示中文标签


def main():
    #Kmeans
    # n_sample = [10,100,1000,10000,60000]
    # times = [0.051, 0.089, 0.794, 8.621, 80.233]
    # NMI = [0.887, 0.601, 0.488, 0.479, 0.49]

    #4.3
    c = [2,4,6,8,10,12,14]
    Je = [492199.66610080487, 449168.3972300443, 419417.11627998087, 402366.5338522838, 388140.30620129814, 376830.09031602496, 367157.9425758028]


    plt.plot(c, Je)
    plt.xlabel('类别数n')
    plt.ylabel('Je')
    plt.show()

    #hierarchical
    # n_sample = [10,100,1000,5000,10000]
    # times = [0.020, 0.055, 0.478,11.667, 49.449]
    # NMI = [0.887, 0.693, 0.543,0.605, 0.66]

    #spectral
    # n_sample = [10,50,100,500,1000]
    # times = [0.045, 0.039, 0.131, 11.958, 63.023]
    # NMI = [0.887, 0.553, 0.534, 0.0663,0.0236]


    # plt.plot(n_sample, times)
    # plt.xlabel('样本数n')
    # plt.ylabel('所需时间s')
    # plt.title('Spectral算法时间复杂度')
    # plt.show()

if __name__ == '__main__':
    main()