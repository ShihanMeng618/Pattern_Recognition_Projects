#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference：https://blog.csdn.net/colourful_sky/article/details/82082854
from myTree import loadData, createTree, test
import numpy as np
from random import randrange
import random

def get_subsample(dataset):
    sub_indexs = []
    length = len(dataset)
    while len(sub_indexs) < length:
        index = randrange(length)
        sub_indexs.append(index)
    return dataset[sub_indexs,:]

def get_n_features(dataset,n_features):
    features = []
    while len(features) < n_features:
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    features.append(len(dataset[0])-1)
    return features
    # return index of n_features

def vote(pre_labels):
    results = []
    for i in range(len(pre_labels[0])): # Nonetype处理一下？ tree中找不到这个值
        col = [line[i] for line in pre_labels]
        col = [np.random.randint(1,10) if x==None else x for x in col]
        result = np.argmax(np.bincount(col))
        results.append(result)
    return results

def main():
    #import data
    X_train, X_val, X_test, y_train, y_val, y_test, keyword = loadData()
    dataSet_train = np.hstack((X_train, y_train))

    #3.1
    # val
    #trees = []
    # accuracys = []
    #
    # for n_tree in [3]:
    #     pre_labels = []
    #     for i in range(n_tree):
    #         featLabels = []
    #         subdataset = get_subsample(dataSet_train)
    #         # n_feature
    #         thresh = 0.1
    #         myTree = createTree(subdataset, keyword, featLabels, thresh=thresh)
    #         pre_label = test(myTree, keyword, featLabels, X_val)
    #         # trees.append(myTree)
    #         pre_labels.append(pre_label)
    #     pre_final_label = vote(pre_labels) # 投票器
    #     accuracy = np.sum(pre_final_label == y_val.reshape(1,-1)[0]) / len(y_val)
    #     accuracys.append(accuracy)
    # print(accuracys)


    # test
    n_tree = 10
    pre_labels = []
    for i in range(n_tree):
        featLabels = []
        subdataset = get_subsample(dataSet_train)
        # n_feature
        thresh = 0.1
        myTree = createTree(subdataset, keyword, featLabels, thresh=thresh)
        pre_label = test(myTree, keyword, featLabels, X_test)
        pre_labels.append(pre_label)

    pre_final_label = vote(pre_labels)  # 投票器
    accuracy = np.sum(pre_final_label == y_test.reshape(1, -1)[0]) / len(y_test)
    print(accuracy)




if __name__ =='__main__':
    main()