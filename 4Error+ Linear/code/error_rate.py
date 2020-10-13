#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from scipy import stats
from random import randint

def calculate_errorrate(mu1, mu2, var1, var2): # 数值计算error
    x, y = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
    xy = np.dstack([x, y])
    pdf1 = stats.multivariate_normal.pdf(xy, mu1, var1)
    pdf2 = stats.multivariate_normal.pdf(xy, mu2, var2)
    error = 0
    for i in range(500):
        for j in range(500):
            if pdf1[i][j] > pdf2[i][j]:
                error += pdf2[i][j]*(6/500)**2 *0.5
            if pdf1[i][j] < pdf2[i][j]:
                error += pdf1[i][j]*(6/500)**2 *0.5
    print("theoretical error rate:")
    print(error)


def cubeWin(u):
    T = abs(u)
    if all(t <= 0.5 for t in T):
        return 1
    else:
        return 0


def gaussWin(u):
    t = 1 / np.sqrt(2*np.pi)
    return t * np.exp(-(u.dot(u))/2)


def parzen(Data, X, h, f) :
    Prob = []
    n = len(Data)
    for x in X :
        p = 0.0
        for sample in Data:
            p += f((sample-x)/h)
        Prob.append(p / (n * (h**2)))
    return np.array(Prob)


def initialize_mixture(data):
    (n,d) = np.shape(data)
    mu = []
    for i in range(2):
        mu.append(data[randint(0,n-1)])
    var = [[1,1]]*2
    pi = [1/2]*2
    y = np.ones((n,2))/2
    return mu, var, pi, y


def getexpectation(data, mu, var, pi):
    (n,d) = np.shape(data)
    pdfs = np.zeros((n,2))
    for i in range(2):
        pdfs[:, i] = pi[i] * stats.multivariate_normal.pdf(data, mu[i], np.diag(var[i])) # 对角矩阵

    y = pdfs / pdfs.sum(axis=1).reshape(-1,1)
    pi = y.sum(axis=0) / y.sum()
    log_likelihood = np.mean(np.log(pdfs.sum(axis=1)))
    return log_likelihood, y, pi


def maximization(data, y):
    (n,d) = np.shape(data)
    mu = np.zeros((2,2))
    var = np.zeros((2,2))
    for i in range(2):
        mu[i] = np.average(data, axis=0, weights=y[:, i])
        var[i] = np.average((data - mu[i])**2, axis=0,weights=y[:, i])
    return mu, var

def main():
    mu1 = [-1,0]
    mu2 = [1,0]
    var1 = np.eye(2)
    var2 = np.array([[2,0],[0,1]])
    pi_1 = pi_2 = 0.5

    # 4.1
    calculate_errorrate(mu1, mu2, var1, var2)

    # 4.2
    # 分别生成n个data,共2n个,分成train，test集
    n = 10000
    data1 = stats.multivariate_normal.rvs(mu1, var1, size = n)
    data1_label = np.insert(data1, 2, values=0, axis=1) #label
    data2 = stats.multivariate_normal.rvs(mu2, var2, size = n)
    data2_label = np.insert(data2, 2, values=1, axis=1) #label
    data = np.vstack([data1_label, data2_label])

    error_rate = []
    for time in range(5): # 重复训练多少次
        print(time)
        # sklearn.model_selection.train_test_split随机划分训练集与测试集。train_test_split(train_data, train_target, test_size=数字,
        #                                                         random_state=0)
        np.random.shuffle(data1_label)
        data1_train = data1_label[0:9000,:]
        data1_test = data1_label[9000:10000,:]
        np.random.shuffle(data2_label)
        data2_train = data2_label[0:9000, :]
        data2_test = data2_label[9000:10000,:]
        data_test = np.vstack([data1_test, data2_test])

        # pNx1 = parzen(data1_train[:, 0:2], data_test[:, 0:2], 1, f=cubeWin)
        # pNx2 = parzen(data2_train[:, 0:2], data_test[:, 0:2], 1, f=cubeWin)

        pNx1 = parzen(data1_train[:,0:2], data_test[:,0:2], 0.5, f=gaussWin)
        pNx2 = parzen(data2_train[:,0:2], data_test[:,0:2], 0.5, f=gaussWin)

        error_cnt = 0
        for index in range(len(pNx1)):
            if pNx1[index] > pNx2[index] and data_test[index, 2] == 1:
                error_cnt += 1
            if pNx1[index] < pNx2[index] and data_test[index, 2] == 0:
                error_cnt += 1
        error_rate.append(error_cnt / 2000)

    print(error_rate)

    # 4.4 EM
    # 采样2n个样本
    # n = 10000
    # data1 = stats.multivariate_normal.rvs(mu1, var1, size=n)
    # data1_label = np.insert(data1, 2, values=0, axis=1)  # label
    # data2 = stats.multivariate_normal.rvs(mu2, var2, size=n)
    # data2_label = np.insert(data2, 2, values=1, axis=1)  # label
    # data = np.vstack([data1_label, data2_label])
    #
    # error_rate = []
    # for time in range(20):  # 重复训练多少次
    #     print(time)
    #     np.random.shuffle(data1_label)
    #     data1_train = data1_label[0:9000,:]
    #     data1_test = data1_label[9000:10000,:]
    #     np.random.shuffle(data2_label)
    #     data2_train = data2_label[0:9000, :]
    #     data2_test = data2_label[9000:10000,:]
    #     data_train = np.vstack([data1_train, data2_train])
    #     data_test = np.vstack([data1_test, data2_test])
    #
    #
    #     [mu, var, pi, y] = initialize_mixture(data_train[:,0:2])
    #     loglh = []
    #     for i in range(5):
    #         [log_likelihood, y, pi] = getexpectation(data_train[:,0:2], mu, var, pi)  # E-step
    #         loglh.append(log_likelihood)
    #         [mu, var] = maximization(data_train[:,0:2], y)  # M-step
    #
    #     pdf1 = stats.multivariate_normal.pdf(data_test[:,0:2], mu[0], np.diag(var[0]))
    #     pdf2 = stats.multivariate_normal.pdf(data_test[:,0:2], mu[1], np.diag(var[1]))
    #
    #     error_cnt = 0
    #     for index in range(len(pdf1)):
    #         if pdf1[index] > pdf2[index] and data_test[index, 2] == 1:
    #             error_cnt += 1
    #         if pdf1[index] < pdf2[index] and data_test[index, 2] == 0:
    #             error_cnt += 1
    #     if error_cnt  > 1000:
    #         error_cnt  = 2000 - error_cnt
    #     error_rate.append(error_cnt / 2000)
    #
    # print(error_rate)


if __name__ == "__main__":
        main()