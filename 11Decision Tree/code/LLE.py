#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference： https://github.com/asdspal/dimRed/blob/master/LLE.ipynb
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D


def knbor_Mat(data, k, dist_metric="euclidean", algorithm="ball_tree"):
    knn = neighbors.NearestNeighbors(k + 1, metric=dist_metric, algorithm=algorithm).fit(data)
    distances, nbors = knn.kneighbors(data)
    return nbors[:, 1:]


def get_weights(data, nbors, reg, k):
    n, p = data.shape
    Weights = np.zeros((n, n))
    for i in range(n):
        data_bors = data[nbors[i], :] - data[i]
        cov_nbors = np.dot(data_bors, data_bors.T)
        # regularization
        trace = np.trace(cov_nbors)
        if trace > 0:
            R = reg * trace
        else:
            R = reg

        cov_nbors.flat[::k + 1] += R
        weights = linalg.solve(cov_nbors, np.ones(k).T, sym_pos=True)

        weights = weights / weights.sum()
        Weights[i, nbors[i]] = weights
    return Weights


def LLE(data, k):
    reg = 0.001
    nbors = knbor_Mat(data, k)
    Weights = get_weights(data, nbors, reg, k)

    n, p = Weights.shape
    m = np.eye(n) - Weights
    M = m.T.dot(m)

    eigvals, eigvecs = np.linalg.eig(M)
    ind = np.argsort(np.abs(eigvals))

    return eigvecs[:, ind[1:3]]

def main():
    # generate data
    num = 2000
    data = np.zeros([num, 3])
    data[:, 2] = np.random.rand(num, 1).ravel()
    data[:, 1] = 3 * np.random.rand(num, 1).ravel()
    data[:, 0] = abs(np.sin(2 * np.pi * data[:, 2]))
    index = np.argsort(data[:, 0], axis=0)
    data[:, 0] = data[index, 0]
    data[:, 2] = data[index, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    c = list(range(num))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=c, cmap='rainbow')
    plt.savefig('2.png')
    plt.show()

    # LLE 结果plot
    plt.figure()
    for k in [40,80]:
        Y = LLE(data , k)
        plt.scatter(Y[:,0],Y[:,1],c=c, cmap='rainbow')
        plt.savefig('LLE_k='+str(k)+'.png')
        plt.show()



if __name__ == '__main__':
    main()