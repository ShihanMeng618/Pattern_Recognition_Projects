#!/usr/bin/python
# -*- coding:utf-8 -*-
# plot the result
from matplotlib import pyplot as plt

def main():
    sample = [100, 300, 1000, 3000, 10000, 30000, 60000]
    k1 = [67.94, 79.23, 86.90, 91.91, 94.63, 96.18, 96.91] #k=1
    k3 = [64.76, 75.90, 86.22, 91.68, 94.63, 96.24, 97.05] #k=3
    k5 = [62.32, 74.46, 85.82, 91.68, 94.42, 96.27, 96.88] #k=5
    k7 = [60.56, 72.96, 85.22, 91.29, 94.43, 96.20, 96.94] #k=7
    k9 = [58.38, 71.46, 84.61, 91.05, 94.08, 96.04, 96.59] #k=9
    # dis k=1
    disL2 = k1
    disL1 = [66.17, 77.55, 85.60, 90.61, 93.67, 95.42, 96.31]
    disL0 = [44.94, 54.88, 64.41, 69.92, 76.30, 80.54, 82.79]

    plt.figure()
    plt.plot(sample, k1, marker='s')
    plt.xlabel("samples")
    plt.ylabel("accuray")
    plt.title("1nn-classifer performance")

    plt.figure()
    p1, = plt.plot(sample, k1, marker='s')
    p2, = plt.plot(sample, k3, marker='s')
    p3, = plt.plot(sample, k5, marker='s')
    p4, = plt.plot(sample, k7, marker='s')
    p5, = plt.plot(sample, k9, marker='s')
    plt.legend((p1,p2,p3,p4,p5), ("k=1", "k=3", "k=5", "k=7", "k=9"), loc='lower right')
    plt.xlabel("samples")
    plt.ylabel("accuray")
    plt.title("knn-classifer performance on various k")

    plt.figure()
    dis1, = plt.plot(sample, disL0, marker='s')
    dis2, = plt.plot(sample, disL1, marker='s')
    dis3, = plt.plot(sample, disL2, marker='s')
    plt.legend((dis3, dis2, dis1), ("L2", "L1", "L0"), loc='lower right')
    plt.xlabel("samples")
    plt.ylabel("accuray")
    plt.title("knn-classifer performance on various distance metrics")

    plt.show()


if __name__ == "__main__":
    main()