#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from pandas import DataFrame

def main():
    # 2D linear classification dataset
    X, y = make_blobs(n_samples=200, centers=[[-2, -2], [1, 1]], n_features=3)
    y = [-1 if label == 0 else label for label in y]
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    df.to_csv("data1.csv")
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.show()

if __name__ == "__main__":
    main()