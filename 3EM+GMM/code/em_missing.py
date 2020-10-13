#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

def missing(data, mu_0, var_0):
    '''
    one time EM estimation
    :param data: x
    :param mu_0: initial mean
    :param var_0: initial covariance
    :return: estimated mean and covariance
    '''
    mu = np.zeros(3) # new
    mu[0:2] = 1 / 10 * data.sum(axis=0)[0:2]

    # set to zero every iteration
    expectation = [] # mean
    mean = 0

    covariance = np.zeros([3, 3]) # covariance
    var = np.zeros([3, 3])
    Dvar = var_0[2, 2] - np.dot(var_0[2, 0:2], np.linalg.inv(var_0[0:2, 0:2])).dot(var_0[0:2, 2])

    for i in range(10):
        if (i + 1) % 2 == 0:  # even num
            # mean
            expectation.append(
                mu_0[2] + np.dot(var_0[2, 0:2], np.linalg.inv(var_0[0:2, 0:2])).dot((data[i, :] - mu_0)[0:2]))
            # covariance
            var[0:2, 0:2] += (data[i, :] - mu_0)[0:2].reshape(-1, 1) * (data[i, :] - mu_0)[0:2]
            var[0, 2] += (data[i, 0] - mu_0[0]) * (expectation[int((i + 1) / 2 - 1)] - mu_0[2])
            var[2, 0] = var[0, 2]
            var[1, 2] += (data[i, 1] - mu_0[1]) * (expectation[int((i + 1) / 2 - 1)] - mu_0[2])
            var[2, 1] = var[1, 2]
            var[2, 2] += Dvar + expectation[int((i + 1) / 2 - 1)] ** 2 + mu_0[2] ** 2 - 2 * mu_0[2] * expectation[
                int((i + 1) / 2 - 1)]
        else:  # odd num
            #mean
            mean += data[i, 2]
            # covariance
            covariance += (data[i, :] - mu_0).reshape(-1, 1) * (data[i, :] - mu_0)
    mu[2] = 1 / 10 * (mean + sum(expectation))
    var = 1 / 10 * (var + covariance)

    return mu, var


def full(data):
    mu = 1/10 * data.sum(axis=0)
    var = 0
    for i in range(10):
        var += (data[i, :] - mu).reshape(-1, 1) * (data[i, :] - mu)
    var = var / 10
    print("在信息完整的情况下：")
    print("估计的均值为"+str(mu))
    print("协方差矩阵为\n"+str(var))
    return mu, var


def main():
    x = np.array([[0.42, -0.087, 0.58],[-0.2, -3.3, -3.4],[1.3, -0.32, 1.7],
                  [0.39, 0.71, 0.23],[-1.6, -5.3, -0.15],[-0.029, 0.89, -4.7],
                  [-0.23, 1.9, 2.2],[0.27, -0.3, -0.87],[-1.9, 0.76, -2.1],[0.87, -1.0, -2.6]])

    # initialize mu， var
    mu_0 = np.zeros(3)
    egvalue = np.ones(3)
    var_0 = np.diag(egvalue)

    # x3 missing
    times = 40 # 保证估计的均值和方差收敛
    for iter in range(times):
        mu,var = missing(x, mu_0,var_0)
        # update
        mu_0 = mu
        var_0 = var

    print("在x3信息缺失的情况下：")
    print("估计的均值为" + str(mu))
    print("协方差矩阵为\n" + str(var))

    # no missing data
    mu_full, var_full = full(x)


if __name__ == "__main__":
    main()