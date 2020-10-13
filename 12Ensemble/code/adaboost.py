# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'  # 用来正常显示中文标签


def adaboost(X, y, X_test, y_test, maxIter):
    '''
    adaboost: carry on adaboost on the data for maxIter loops
    Input 
        X       : n * p matirx, training data
        y       : (n, ) vector, training label
        X_test  : m * p matrix, testing data
        y_test  : (m, ) vector, testing label
        maxIter : number of loops
    Output
        e_train : (maxIter, ) vector, errors on training data
        e_test  : (maxIter, ) vector, errors on testing data
    '''

    w = np.ones(y.shape, dtype='float') / y.shape[0]

    k = np.zeros(maxIter, dtype='int')
    a = np.zeros(maxIter)
    d = np.zeros(maxIter)
    alpha = np.zeros(maxIter)

    e_train = np.zeros(maxIter)
    e_test = np.zeros(maxIter)

    for i in range(maxIter):
        k[i], a[i], d[i] = decision_stump(X, y, w)
        print('new decision stump k:%d a:%f, d:%d' % (k[i], a[i], d[i]))
        
        e = decision_stump_error(X, y, k[i], a[i], d[i], w)
        alpha[i] = np.log((1 - e) / e)
        w = update_weights(X, y, k[i], a[i], d[i], w, alpha[i])
        
        e_train[i] = adaboost_error(X, y, k, a, d, alpha)
        e_test[i] = adaboost_error(X_test, y_test, k, a, d, alpha)
        print('weak learner error rate: %f\nadaboost error rate: %f\ntest error rate: %f\n' % (e, e_train[i], e_test[i]))

    return e_train, e_test


def decision_stump(X, y, w):
    '''
    decision_stump returns a rule ...
    h(x) = d if x(k) <= a, −d otherwise,
    Input
        X : n * p matrix, each row a sample
        y : (n, ) vector, each row a label
        w : (n, ) vector, each row a weight
    Output
        k : the optimal dimension
        a : the optimal threshold
        d : the optimal d, 1 or -1
    '''

    # total time complexity required to be O(p*n*logn) or less
    D = [1,-1]
    n, p = X.shape
    X_max = np.max(X)
    X_min = np.min(X)
    step = 0.1 # 可变
    num = int((X_max-X_min)/step)
    e = np.zeros([2,num])
    error = np.zeros(p)
    D1 = np.zeros(p)
    thre = np.zeros(p)
    for i in range(p):
        for j in range(2):
            d = D[j]
            for k in range(num):
                a = X_min + k*step
                e[j,k] = decision_stump_error(X, y, i, a, d, w)
        index = np.argsort(e, axis=0) # 对列升序排序
        e = np.sort(e, axis=0)
        index_e = np.argmin(e[0,:])
        error[i] = e[0,index_e]
        D1[i] = D[index[0,index_e]]
        thre[i] = X_min + index_e * step
    k = np.argmin(error)
    d = D1[k]
    a = thre[k]

    return k, a, d


def decision_stump_error(X, y, k, a, d, w):
    '''
    decision_stump_error returns error of the given stump
    Input
        X : n * p matrix, each row a sample
        y : (n, ) vector, each row a label
        k : selected dimension of features
        a : selected threshold for feature-k
        d : 1 or -1
    Output
        e : number of errors of the given stump
    '''
    p = ((X[:, k] <= a).astype('float') - 0.5) * 2 * d # predicted label
    e = np.sum((p.astype('int') != y) * w)

    return e


def update_weights(X, y, k, a, d, w, alpha):
    '''
    update_weights update the weights with the recent classifier

    Input
        X        : n * p matrix, each row a sample
        y        : (n, ) vector, each row a label
        k        : selected dimension of features
        a        : selected threshold for feature-k
        d        : 1 or -1
        w        : (n, ) vector, old weights
        alpha    : weights of the classifiers

    Output
        w_update : (n, ) vector, the updated weights
    '''

    p = ((X[:, k] <= a).astype('float') - 0.5) * 2 * d # predicted label
    w = w * np.exp(-0.5 * alpha * y * p)
    w_update = w / np.sum(w)

    return w_update


def adaboost_error(X, y, k, a, d, alpha):
    '''
    adaboost_error: returns the final error rate of a whole adaboost
    
    Input
        X     : n * p matrix, each row a sample
        y     : (n, ) vector, each row a label
        k     : (iter, ) vector,  selected dimension of features
        a     : (iter, ) vector, selected threshold for feature-k
        d     : (iter, ) vector, 1 or -1
        alpha : (iter, ) vector, weights of the classifiers
    Output
        e     : error rate  
    '''

    n, p = X.shape
    iter = len(k)
    pre_label = []
    for j in range(n):
        temp = 0
        for i in range(iter):
            p = ((X[j, k[i]] <= a[i]).astype('float') - 0.5) * 2 * d[i]
            temp += 0.5 * alpha[i] * p
        pre_label.append(np.sign(temp))
    e = 1 - np.sum(pre_label==y) / n
    return e


def main():
    dataFile = 'ada_data.mat'
    data = scio.loadmat(dataFile)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train'].ravel()
    y_test = data['y_test'].ravel()

    e_train, e_test = adaboost(X_train, y_train, X_test, y_test, 300)
    plt.figure()
    plt.plot(e_train, label = 'e_train')
    plt.plot(e_test, label = 'e_test')
    plt.legend(loc = 'upper right')
    plt.title('训练集和测试集随迭代次数的变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('错误率')
    plt.savefig('adaboost01.png')
    plt.show()

if __name__ == '__main__':
    main()
