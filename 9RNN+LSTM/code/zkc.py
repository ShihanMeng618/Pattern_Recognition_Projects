#!/usr/bin/python
# -*- coding:utf-8 -*-
from networkx import karate_club_graph, to_numpy_matrix
import numpy as np
import matplotlib.pyplot as plt

def relu(X):
    return np.maximum(0,X)

def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)

def main():
    zkc = karate_club_graph()
    order = sorted(list(zkc.nodes()))
    A = to_numpy_matrix(zkc, nodelist=order)
    I = np.eye(zkc.number_of_nodes())
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.asmatrix(np.diag(D_hat))

    W_1 = np.random.normal(size=(zkc.number_of_nodes(), 4))
    W_2 = np.random.normal(size=(W_1.shape[1], 2))

    H_1 = gcn_layer(A_hat, D_hat, I, W_1)
    H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
    output = H_2

    feature_representations = {node: np.array(output)[node] for node in zkc.nodes()}

    plt.figure()
    for i in range(34):
        if zkc.nodes[i]['club'] == 'Mr. Hi':
            plt.scatter(np.array(output)[i,0], np.array(output)[i,1],color = 'b',alpha=0.5,s = 100)
        else:
            plt.scatter(np.array(output)[i,0], np.array(output)[i,1],color = 'r',alpha=0.5,s = 100)
    plt.show()


if __name__ == '__main__':
    main()
