#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference: https://scikit-learn.org/stable/modules/svm.html#svm-classification
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm


def main():
    x = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1],[-0.1, 5.2],[-4.0, 2.2],
                  [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1], #--#
                  [-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
                  [-0.5, 9.2], [-5.3, 6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])
    y = np.array([0]*10 + [1]*10)

    #fit the model
    clf = svm.SVC(kernel='poly', degree=2, coef0=1.0)
    for num in range(1,11): # repeat
        print(num)
        x_train = np.vstack([x[0:num], x[10:(10+num)]])
        y_train = np.hstack([y[0:num], y[10:(10+num)]])
        clf.fit(x_train, y_train)

        # create a mesh to plot
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # plot the line, the points, and the nearest vectors to the plane
        plt.clf()
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    zorder=10, facecolors='none', edgecolors='k') # support_vectors
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, zorder=10, s=20, edgecolors='k') # train data

        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) # decision function
        margin = clf.decision_function(clf.support_vectors_)
        print(margin)

        # Put the result into a color plot
        z = z.reshape(xx.shape)
        colours = (["bisque", "lightskyblue"])
        cmap = ListedColormap(colours)
        plt.pcolormesh(xx, yy, z > 0, cmap=cmap)
        plt.contour(xx, yy, z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        plt.show()

if __name__ == "__main__":
    main()