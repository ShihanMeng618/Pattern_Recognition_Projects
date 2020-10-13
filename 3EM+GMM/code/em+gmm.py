#!/usr/bin/python
# -*- coding:utf-8 -*-
# 参考代码 http://sofasofa.io/tutorials/gmm_em/

import scipy.io as scio
from scipy.stats import multivariate_normal
import numpy as np
import math
from random import randint
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

def initialize_mixture(data, m):
    """
    initialize GMM
    :param data:  x
    :param m: num of different Gaussian distribution
    :return: mean, covariance, pi, hidden variable matrix y
    """
    (n,d) = np.shape(data)
    mu = []
    for i in range(m):
        mu.append(data[randint(0,n-1)])# 随机选择point作为mu
    var = [[1,1]]*m
    pi = [1/m]*m
    y = np.ones((n,m))/m
    return mu, var, pi, y

def getexpectation(data, mu, var,pi):
    """
    E-step
    :param data: x
    :param mu: mean
    :param var: covariance
    :param pi:
    :return: log_likelihood, y, new pi
    """
    (n,d) = np.shape(data)
    m = len(pi)
    pdfs = np.zeros((n,m))
    for i in range(m):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(data, mu[i], np.diag(var[i])) # 对角矩阵

    y = pdfs / pdfs.sum(axis=1).reshape(-1,1)
    pi = y.sum(axis=0) / y.sum()
    log_likelihood = np.mean(np.log(pdfs.sum(axis=1)))
    return log_likelihood, y, pi


def plot_clusters(data, mu, var):
     m = len(mu)
     plt.figure()
     plt.scatter(data[:,0], data[:,1])
     ax = plt.gca()
     for i in range(m):
         ellipse = Ellipse(mu[i], var[i][0], var[i][1], fc ='None', lw = 2, color='r')
         ax.add_patch(ellipse)
     plt.show()


def maximization(data, y):
    """
    M-step
    :param data: x
    :param y: hidden variable
    :return: new mean and new covariance
    """
    (n,d) = np.shape(data)
    m = y.shape[1]
    mu = np.zeros((m,2))
    var = np.zeros((m,2))
    for i in range(m):
        mu[i] = np.average(data, axis=0, weights=y[:, i])
        var[i] = np.average((data - mu[i])**2, axis=0,weights=y[:, i])
    # modified M-step
    temp = np.average(var, axis=0)
    for i in range(m):
        var[i] = temp
    return mu, var


def main():
    # import data x emdata.m文件读入并画图
    data_path = "emdata.mat"
    data = scio.loadmat(data_path)
    data_x = data["data2"] # x=1000*2
    times = 5 # 迭代次数，保证log_likelihood收敛

    for m in [2,3,4,5]:
        print(m)
        [mu, var, pi, y] = initialize_mixture(data_x, m)
        loglh = []
        for i in range(times):
            [log_likelihood, y, pi] = getexpectation(data_x, mu, var, pi) # E-step
            loglh.append(log_likelihood)
            [mu, var] = maximization(data_x, y) # M-step
            print('log-likelihood:%.3f'%loglh[-1])
            if i == times-1:
                plot_clusters(data_x, mu, var) # 画图
                # BIC = -math.log(1000) * m + 2*log_likelihood # BIC准则选择模型
                # print(BIC)


if __name__ == "__main__":
    main()