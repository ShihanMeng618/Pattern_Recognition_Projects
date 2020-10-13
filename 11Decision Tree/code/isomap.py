#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances


def shortest_path(dist):
    num = len(dist)
    iter = 0
    for k in range(num):
        for i in range(num):
            for j in range(num):
                if dist[i][j] > dist[i][k]+dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def main():
    # generate data
    num = 2000
    data = np.zeros([num,3])
    data[:,0] = np.random.rand(num,1).ravel()
    data[:,1] = 2 * np.random.rand(num,1).ravel()
    data[:,2] = -np.cos(3*np.pi*data[:,0])
    index = np.argsort(data[:,0],axis=0)
    data[:,0] = data[index,0]
    data[:,2] = data[index,2]
    fig = plt.figure()
    ax = Axes3D(fig)
    c = list(range(num))
    ax.scatter(data[:,1],data[:,2], data[:,0], c=c,cmap='rainbow')
    plt.savefig('1.png')
    plt.show()


    for k in [10,30,50,100]:
        # knn
        dist_matrix = pairwise_distances(data, metric="euclidean") # num * num
        inf = np.max(dist_matrix)*100
        knn_matrix = np.ones([num,num]) * inf
        arg = np.argsort(dist_matrix, axis=1)
        for i in range(num):
            knn_matrix[i, arg[i, 0:k+1]] = dist_matrix[i, arg[i, 0:k+1]]
        knn_matrix = shortest_path(knn_matrix)
        print(knn_matrix.max())

        # mds
        A = -1 / 2 * knn_matrix * knn_matrix
        H = np.eye(num) - 1 / num * np.dot(np.ones(num).reshape(-1, 1), np.ones(num).reshape(1, -1))
        B = H.dot(A).dot(H)
        eig_lambda, eig_vects = np.linalg.eig(B)
        index = eig_lambda.argsort()[::-1]
        vects = eig_vects[:, index]  # 按lambda降序对vectors排序
        X = vects[:, 0:2].dot(np.sqrt(np.diag(eig_lambda[0:2])))
        plt.scatter(X[:,0],X[:,1],c=c, cmap='rainbow')
        plt.savefig('k='+str(k)+'.png')
        plt.show()


if __name__ == "__main__":
    main()