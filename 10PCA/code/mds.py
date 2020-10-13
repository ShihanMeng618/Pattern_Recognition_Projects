#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'  # 用来正常显示中文标签

def main():
    D = np.array([[0,18,10,7,10,13.5,44,2],
                 [18,0,19,18,27,17.5,31.5,31],
                 [10,19,0,11.5,18.5,11,45.5,39],
                 [7,18,11.5,0,25.5,33.5,26,9],
                 [10,27,18.5,25.5,0,22.5,52.5,38],
                 [13.5,17.5,11,33.5,22.5,0,43.5,50],
                 [44,31.5,45.5,26,52.5,43.5, 0, 43],
                 [12,31,39,9,38,50,43,0]])
    label = [u'北京',u'甘肃天水',u'山西五台',u'湖北武汉',u'黑龙江哈尔滨',u'宁夏永宁',u'四川九龙',u'广东广州']
    D_sym = 1/2 *(D+D.T)
    n = len(D)
    A = -1/2 * D_sym * D_sym
    H = np.eye(n) - 1/n * np.dot(np.ones(n).reshape(-1,1), np.ones(n).reshape(1,-1))
    B = H.dot(A).dot(H)
    eig_lambda, eig_vects = np.linalg.eig(B)
    index = eig_lambda.argsort()[::-1]
    vects = eig_vects[:,index]  # 按lambda降序对vectors排序
    X = vects[:,0:2].dot(np.sqrt(np.diag(eig_lambda[0:2])))
    for i in range(n):
        plt.scatter(X[i,0], X[i,1])
        plt.text(X[i,0], X[i,1], label[i])
    plt.title('MDS恢复结果')
    plt.show()


if __name__ == "__main__":
    main()