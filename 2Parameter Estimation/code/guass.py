#!/usr/bin/python
# -*- coding:utf-8 -*-
# This program implements the Parzen window method
# in non-parametric estimation by Gaussian window
import numpy as np
from matplotlib import pyplot as plt


def GaussWin(N,a,x):
    '''
    Guassian window function
    :param N: number of samples
    :param a: window width
    :param x: sequence xi
    :return: estimation result pNx
    '''
    hN = a / np.sqrt(N)
    pNx = np.zeros(10000)
    for j in range(10000):
        for i in range(N):
            pNx[j] += np.exp(((x[j]-x[i])/hN)*((x[j]-x[i])/hN)/-2)/ np.sqrt(2 * np.pi) / hN
        pNx[j] /= N
    return pNx


def expNcal(pNx,y1):
    '''
    compute e(pn) = intergate [pn(x) − p(x)]^2dx numerically
    :param pNx: estimation result pNx
    :param y1: p(x)
    :return:
    '''
    expN = 0
    for i in range(10000):
        expN += (pNx[i] - y1[i])*(pNx[i] - y1[i])
    return expN

def expN_exp_var(times):
    '''
    compute the expectation and variance of e(pn) w.r.t different n and a
    N = 5,10,50,100,500,1000,5000,10000
    a = 0.25,1,4
    :param times: num of iteration
    :return: the matrix of e(pn) with different n and a
    '''
    expN = np.zeros([times, 24])

    for i in range(times):
        print(i)
        x1 = np.random.normal(-1, 1, 10000)
        x2 = np.random.normal(1, 1, 10000)
        x = 0.2 * x1 + 0.8 * x2
        one = np.ones(10000)
        y1 = 0.2 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 * (x + one) * (x + one) / 2) + 0.8 * 1 / np.sqrt(
            2 * np.pi) * np.exp(-1 * (x - one) * (x - one) / 2)

        pNx0 = GaussWin(5, 0.25, x)
        expN[i, 0] = expNcal(pNx0, y1)

        pNx1 = GaussWin(10, 0.25, x)
        expN[i, 1] = expNcal(pNx1, y1)

        pNx2 = GaussWin(50, 0.25, x)
        expN[i, 2] = expNcal(pNx2, y1)

        pNx3 = GaussWin(100, 0.25, x)
        expN[i, 3] = expNcal(pNx3, y1)

        pNx4 = GaussWin(500, 0.25, x)
        expN[i, 4] = expNcal(pNx4, y1)

        pNx5 = GaussWin(1000, 0.25, x)
        expN[i, 5] = expNcal(pNx5, y1)

        pNx6 = GaussWin(5000, 0.25, x)
        expN[i, 6] = expNcal(pNx6, y1)

        pNx7 = GaussWin(10000, 0.25, x)
        expN[i, 7] = expNcal(pNx7, y1)

        pNx0 = GaussWin(5, 1, x)
        expN[i, 8] = expNcal(pNx0, y1)

        pNx1 = GaussWin(10, 1, x)
        expN[i, 9] = expNcal(pNx1, y1)

        pNx2 = GaussWin(50, 1, x)
        expN[i, 10] = expNcal(pNx2, y1)

        pNx3 = GaussWin(100, 1, x)
        expN[i, 11] = expNcal(pNx3, y1)

        pNx4 = GaussWin(500, 1, x)
        expN[i, 12] = expNcal(pNx4, y1)

        pNx5 = GaussWin(1000, 1, x)
        expN[i, 13] = expNcal(pNx5, y1)

        pNx6 = GaussWin(5000, 1, x)
        expN[i, 14] = expNcal(pNx6, y1)

        pNx7 = GaussWin(10000, 1, x)
        expN[i, 15] = expNcal(pNx7, y1)

        pNx0 = GaussWin(5, 4, x)
        expN[i, 16] = expNcal(pNx0, y1)

        pNx1 = GaussWin(10, 4, x)
        expN[i, 17] = expNcal(pNx1, y1)

        pNx2 = GaussWin(50, 4, x)
        expN[i, 18] = expNcal(pNx2, y1)

        pNx3 = GaussWin(100, 4, x)
        expN[i, 19] = expNcal(pNx3, y1)

        pNx4 = GaussWin(500, 4, x)
        expN[i, 20] = expNcal(pNx4, y1)

        pNx5 = GaussWin(1000, 4, x)
        expN[i, 21] = expNcal(pNx5, y1)

        pNx6 = GaussWin(5000, 4, x)
        expN[i, 22] = expNcal(pNx6, y1)

        pNx7 = GaussWin(10000, 4, x)
        expN[i, 23] = expNcal(pNx7, y1)

    return expN


def drawGauss(a):
    '''
    draw pn(x) under different n and a and p(x) to show the estimation effect
    :param a: window width
    :return: None
    '''
    x1 = np.random.normal(-1, 1, 10000)
    x2 = np.random.normal(1, 1, 10000)
    x = 0.2 * x1 + 0.8 * x2
    one = np.ones(10000)
    y1 = 0.2 * 1 / np.sqrt(2 * np.pi) * np.exp(-1*(x+one)*(x+one)/2)+ 0.8 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 *(x-one)*(x-one) / 2)

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(x, y1, '.')
    plt.ylim([0,0.8])
    plt.title('p(x)')

    plt.subplot(3, 3, 2)
    pNx = GaussWin(5, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=5,a='+str(a))

    plt.subplot(3, 3, 3)
    pNx = GaussWin(10, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=10,a='+str(a))

    plt.subplot(3, 3, 4)
    pNx = GaussWin(50, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=50,a='+str(a))

    plt.subplot(3, 3, 5)
    pNx = GaussWin(100, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=100,a='+str(a))

    plt.subplot(3, 3, 6)
    pNx = GaussWin(500, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=500,a='+str(a))

    plt.subplot(3, 3, 7)
    pNx = GaussWin(1000, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=1000,a='+str(a))

    plt.subplot(3, 3, 8)
    pNx = GaussWin(5000, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=5000,a='+str(a))

    plt.subplot(3, 3, 9)
    pNx = GaussWin(10000, a, x)
    plt.plot(x, pNx, '.')
    plt.ylim([0,0.8])
    plt.title('N=10000,a='+str(a))

def main():
    #(a) GaussWin
    # drawGauss(0.25)
    # drawGauss(1)
    # drawGauss(4)
    # plt.show()

    # (b)(c)
    times = 100
    expN = expN_exp_var(times)
    expectation = np.mean(expN,0)
    variance = np.var(expN,0)

    np.vstack((expN,expectation,variance))
    np.savetxt('gauss_100.csv', expN, delimiter=',')
    print('expectation:'+str(expectation) + 'variance:' + str(variance))


if __name__ == "__main__":
        main()