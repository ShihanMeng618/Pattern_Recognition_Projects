#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference：https://blog.csdn.net/colourful_sky/article/details/82082854
from myTree import loadData, test, calcShannonEnt, splitDataSet, majorityCnt
import numpy as np
import copy
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
        if index not in features: # no replacement
            features.append(index)
    return features
    # return index of n_features


def chooseBestFeatureToSplit(dataset, n_features):
    #特征数量
    numFeatures = n_features
    sub_features_index = get_n_features(dataset, n_features)
    samples_dataset = np.hstack((np.array(dataset)[:,sub_features_index],np.array(dataset)[:,-1].reshape(-1,1))) # 带label
    #计数数据集的香农熵
    baseEntropy = calcShannonEnt(samples_dataset)
    #信息增益
    bestInfoGain = 0.0
    #最优特征的索引值
    bestFeature = -1
    #遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in samples_dataset]
        #创建set集合{}，元素不可重复
        uniqueVals = set(featList)
        #经验条件熵
        newEntropy = 0.0
        #计算信息增益
        for value in uniqueVals:
            #subDataSet划分后的子集
            subDataSet = splitDataSet(samples_dataset, i, value)
            #计算子集的概率
            prob = len(subDataSet) / float(len(samples_dataset))
            #根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        #信息增益
        infoGain = baseEntropy - newEntropy
        #打印每个特征的信息增益
        if i%100 == 0:
            print(i)
            #print("第%d个特征的增益为%.3f" % (i, infoGain))
        #计算信息增益
        if (infoGain > bestInfoGain):
            #更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            #记录信息增益最大的特征的索引值
            bestFeature = i
            #返回信息增益最大特征的索引值
    return sub_features_index[bestFeature]


def createTree(dataset, keyword, featLabels, thresh, n_features):
    #取分类标签, 14400*1, 0-9
    if len(dataset) <= 2:
        return
    classList = [example[-1] for example in dataset]
    #如果满足条件，则停止继续划分
    #遍历完所有特征时返回出现次数最多的类标签
    max_class, cnt = majorityCnt(classList)
    if cnt/len(classList) > 1 - thresh or cnt <= 10 :
        return max_class

    #选择最优特征index
    bestFeat = chooseBestFeatureToSplit(dataset, n_features=n_features)
    #最优特征的标签
    bestFeatLabel = keyword[bestFeat] # str
    featLabels.append(bestFeatLabel)
    #根据最优特征的标签生成树
    myTree={bestFeatLabel:{}}
    #删除已经使用的特征标签
    keyword = np.delete(keyword,bestFeat)
    #del(keyword[bestFeat])
    #属性值
    uniqueVls=[0,1]
    #遍历特征，创建决策树
    for value in uniqueVls:
        keywords = copy.deepcopy(keyword)
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataset, bestFeat, value),
                                               keywords, featLabels,thresh, n_features)
    return myTree

def vote(pre_labels):
    results = []
    for i in range(len(pre_labels[0])):
        col = [line[i] for line in pre_labels]
        col = [np.random.randint(1,10) if x==None else x for x in col]
        result = np.argmax(np.bincount(col))
        results.append(result)
    return results

def main():
    #import data
    X_train, X_val, X_test, y_train, y_val, y_test, keyword = loadData()
    dataSet_train = np.hstack((X_train, y_train.astype('int')))

    #3.2 random features
    # n_tree = 3
    # accuracys = []
    # for n_features in [35]:
    #     pre_labels = []
    #     for i in range(n_tree):
    #         subsamples = get_subsample(dataSet_train)
    #         featLabels = []
    #         thresh = 0.1
    #         myTree = createTree(subsamples, keyword, featLabels, thresh=thresh, n_features=n_features)
    #         pre_label = test(myTree, keyword, featLabels, X_val)
    #         # trees.append(myTree)
    #         pre_labels.append(pre_label)
    #     pre_final_label = vote(pre_labels)  # 投票器
    #     accuracy = np.sum(pre_final_label == y_val.reshape(1, -1)[0]) / len(y_val)
    #     accuracys.append(accuracy)
    # print(accuracys)

    # test
    n_tree = 10
    n_features = 35
    pre_labels = []
    accuracys = []
    accuracys_of_individual_trees = []
    for iter in range(1): # 重复几次试验
        for i in range(n_tree):
            featLabels = []
            subdataset = get_subsample(dataSet_train)
            thresh = 0.1
            myTree = createTree(subdataset, keyword, featLabels, thresh=thresh, n_features=n_features)
            pre_label = test(myTree, keyword, featLabels, X_test)
            pre_labels.append(pre_label)
            acc = np.sum(pre_label == y_test.reshape(1, -1)[0]) / len(y_test)
            accuracys_of_individual_trees.append(acc)

        pre_final_label = vote(pre_labels)  # 投票器
        accuracy = np.sum(pre_final_label == y_test.reshape(1, -1)[0]) / len(y_test)
        accuracys.append(accuracy)

    print(accuracys_of_individual_trees)
    print(accuracys)


if __name__ =='__main__':
    main()