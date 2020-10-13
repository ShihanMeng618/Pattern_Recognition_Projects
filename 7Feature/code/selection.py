#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt


def least_sq_L1(X, y, Lambda, w_0):
    '''
    :param X: n * M matrix, each row a sample with M features
    :param y: n * 1 vector, each element a target
    :param Lambda: the penalty constant
    :param w_0: M * 1 vector, the initial weight
    :return: M * 1 vector, the weight vector
    '''
    [n, M] = np.shape(X)
    a = sum(X**2) / n #(M,1)
    w = w_0
    iter = 0
    err_tol = 1e-8
    err = 1
    while err > err_tol:
        for k in range(M):
            # evaluate c_k based on w and X
            X2 = X[:, k]
            w1 = np.delete(w, k, 0) # (M-1, 1)
            X1 = np.delete(X, k, 1) # (n, M-1)
            c_k = np.dot(X2, y-np.dot(X1, w1)).item() / n
            J = 0.5 * np.dot((y - np.dot(X, w)).reshape(1, -1), y - np.dot(X, w)) / n + Lambda * sum(abs(w))

            # update w[k]
            b = np.array([c_k + Lambda, c_k - Lambda])
            e = b[0] * b[1]
            index = np.argmin(abs(b))
            w[k]= (e > 0) * b[index] / a[k]
            J1 = 0.5 * np.dot((y - np.dot(X, w)).reshape(1, -1), (y-np.dot(X, w))) / n + Lambda * sum(abs(w))

        err = abs(J1 - J)
        iter = iter + 1
    return w


def least_sq_multi(X, y, Lambda, w_0):
    '''
    :param X: n * M matrix, each row a sample with M features
    :param y: n * 1 vector, each element a target
    :param Lambda: 1 * L vecotr, each element a L1-norm penalty constant
    :param w_0: M * 1 vector, the initial weight
    :return: M * L matrix, the column a weight vector
    '''
    [_, M] = np.shape(X)
    L = len(Lambda)
    W = np.zeros([M, L])

    w_l = w_0
    for l in range(L):
        w_l = least_sq_L1(X, y, Lambda[l], w_l)
        W[:, l] = w_l.reshape(1,-1)

    return W


def main():
    # Step 1: Data preprocessing
    data_path = "least_sq.mat"
    data = scio.loadmat(data_path)
    test = data['test']
    test_X = test['X'].any() # (1024,16)
    test_y = test['y'].any() # (1024, 1)
    (n_test, M) = np.shape(test_X)

    # train = data["train_small"] # (8,16)
    # train = data["train_mid"] # (16,16)
    train = data["train_large"] # (64,16)
    train_X = train['X'].any()
    train_y = train['y'].any()
    (n, M) = np.shape(train_X)

    Lambda = np.linspace(0.01, 2.0, 200)
    w_0 = np.linalg.inv(train_X.T.dot(train_X)).dot(train_X.T.dot(train_y))

    # Step 2: Train weight vectors with different penalty constants
    W = least_sq_multi(train_X, train_y, Lambda, w_0) # each column a weight vector

    # Step 3: plot different errors versus lambda
    L = len(Lambda)
    err_Lambda = np.zeros([L, 5]) # each row a different lambda
    for l in range(L):
        w = W[:, l]
        # training error multiplying 1 / 2
        temp1 = train_y - np.dot(train_X,w).reshape(-1,1)
        err_Lambda[l, 0] = 0.5 * np.dot(temp1.reshape(1,-1), temp1) / n

        # L1 regularization penalty
        err_Lambda[l, 1] = sum(abs(w))

        # minimized objective
        err_Lambda[l, 2] = err_Lambda[l, 0] + Lambda[l] * err_Lambda[l, 1]

        # L0 norm: non-zero parameters
        err_Lambda[l, 3] = np.linalg.norm(w, 0)

        # test error
        temp2 = test_y - np.dot(test_X, w).reshape(-1,1)
        err_Lambda[l, 4] = 0.5 * np.dot(temp2.reshape(1,-1), temp2) / n_test

    # Step 4: plot the results
    plt.figure()
    plt.plot(Lambda, err_Lambda[:, 0])
    plt.xlabel('lambda')
    plt.ylabel('training error')
    plt.title('training error vs lambda')

    plt.figure()
    plt.plot(Lambda, err_Lambda[:, 1])
    plt.xlabel('lambda')
    plt.ylabel('L1 penalty')
    plt.title('L1 regularization penalty vs lambda')

    plt.figure()
    plt.plot(Lambda, err_Lambda[:, 2])
    plt.xlabel('lambda')
    plt.ylabel('objective')
    plt.title('objective vs lambda')

    plt.figure()
    plt.plot(Lambda, err_Lambda[:, 3])
    plt.xlabel('lambda')
    plt.ylabel('number features')
    plt.title('number features vs lambda')

    plt.figure()
    plt.plot(Lambda, err_Lambda[:, 4])
    plt.xlabel('lambda')
    plt.ylabel('test error')
    plt.title('test error vs lambda')
    plt.show()

    test_err = err_Lambda[:,4]
    min_index = np.argmin(test_err)
    print('Lambda = ' + str(Lambda[min_index]) + 'error= ' + str(test_err[min_index]))

if __name__ == '__main__':
    main()
