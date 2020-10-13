#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def Single_sample(X, label, gamma):
    yk = np.zeros([200, 3])
    yk[:, 0] = 1
    yk[:, 1:3] = X
    for i in range(3):
        yk[:, i] = yk[:, i] * label

    alpha = np.array([0, 0, 0])  # 行向量
    k = 0
    while k < 200:
        if alpha.dot(yk[k, :]) <= gamma:
            alpha = alpha + yk[k, :]
            k = 0
        else:
            k = k + 1

    return alpha


def main():
    # data preprocessing
    read_csv = pd.read_csv("data.csv")
    X = np.zeros([200, 2])
    X[:,0] = read_csv['x']
    X[:,1] = read_csv['y']
    label = read_csv['label']

    alpha_0 = Single_sample(X, label, 0)
    x = np.linspace(-4, 3, 100)
    y0 = (-alpha_0[1]*x - alpha_0[0])/alpha_0[2]

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=label)
    plt.plot(x, y0)

    alpha = [0]*2
    y = [0]*2
    for index in range(2):
        alpha[index] = Single_sample(X, label, np.power(10, index))
        x = np.linspace(-4, 3, 100)
        y[index] = (-alpha[index][1]*x - alpha[index][0])/alpha[index][2]
        plt.plot(x, y[index])

    plt.show()


if __name__ == "__main__":
    main()