#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference：https://blog.csdn.net/jiaoyangwm/article/details/79525237
import numpy as np
from math import log
import scipy.io as scio
import pandas as pd
import operator
import copy
from sklearn.model_selection import train_test_split


def loadData():
     data = scio.loadmat("Sogou_data/Sogou_webpage.mat")
     word_mat = data['wordMat'] # 14400*1200 doc_num * keyword_num 0-1矩阵
     doc_label = data['doclabel']# 14400*1 每篇文档对应的标签 0-9
     data_frame = pd.read_excel('Sogou_data/keyword.xls')
     keyword = data_frame.values[:,1] # 1200个关键词
     # 3:1:1 train:validation:test
     X_train, X_test, y_train, y_test = train_test_split(word_mat, doc_label, test_size=0.2)
     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
     return X_train, X_val, X_test, y_train, y_val, y_test, keyword


def calcShannonEnt(dataset):
    #返回数据集行数
    numEntries=len(dataset)
    #保存每个标签（label）出现次数的字典
    labelCounts={}
    #对每组特征向量进行统计
    for label in dataset:
        currentLabel = label[-1]                 #提取标签信息
        if currentLabel not in labelCounts.keys():   #如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1                 #label计数

    shannonEnt=0.0                                   #经验熵
    #计算经验熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries      #选择该标签的概率
        shannonEnt -= prob*log(prob,2)                 #利用公式计算
    return shannonEnt                                #返回经验熵


def chooseBestFeatureToSplit(dataset):
    #特征数量
    numFeatures = len(dataset[0]) - 1
    #计数数据集的香农熵
    baseEntropy = calcShannonEnt(dataset)
    #信息增益
    bestInfoGain = 0.0
    #最优特征的索引值
    bestFeature = -1
    #遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataset]
        #创建set集合{}，元素不可重复
        uniqueVals = set(featList)
        #经验条件熵
        newEntropy = 0.0
        #计算信息增益
        for value in uniqueVals:
            #subDataSet划分后的子集
            subDataSet = splitDataSet(dataset, i, value)
            #计算子集的概率
            prob = len(subDataSet) / float(len(dataset))
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
    return bestFeature


def splitDataSet(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet: #1*1201
        if featVec[axis] == value:
            #reducedFeatVec = featVec[:axis]
            #reducedFeatVec.extend(featVec[axis+1:])
            reducedFeatVec = np.delete(featVec, axis)
            retDataSet.append(reducedFeatVec)
            #retDataSet = np.vstack((retDataSet, reducedFeatVec))
    return retDataSet


def majorityCnt(classList):
    classCount={}
    #统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #根据字典的值降序排列
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0], sortedClassCount[0][1]


def createTree(dataset, keyword, featLabels, thresh):
    #取分类标签, 14400*1, 0-9
    if len(dataset) == 0:
        return
    classList = [example[-1] for example in dataset]
    #如果满足条件，则停止继续划分
    #if classList.count(classList[0]) == len(classList):
        #return classList[0]
    #遍历完所有特征时返回出现次数最多的类标签
    max_class, cnt = majorityCnt(classList)
    if cnt/len(classList) > 1 - thresh:# or len(dataset[0]) == 1:
        return max_class

    #选择最优特征index
    bestFeat = chooseBestFeatureToSplit(dataset)
    #最优特征的标签
    bestFeatLabel = keyword[bestFeat] # str
    featLabels.append(bestFeatLabel)
    #根据最优特征的标签生成树
    myTree={bestFeatLabel:{}}
    #删除已经使用的特征标签
    keyword = np.delete(keyword,bestFeat)
    #del(keyword[bestFeat])
    #得到训练集中所有最优特征的属性值
    #featValues=[example[bestFeat] for example in dataSet]
    #去掉重复的属性值
    uniqueVls=[0,1]
    #遍历特征，创建决策树
    for value in uniqueVls:
        keywords = copy.deepcopy(keyword)
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataset, bestFeat, value),
                                               keywords, featLabels,thresh)
    return myTree


def classify(inputTree, featLabels, testVec):
    #获取决策树节点
    firstStr = next(iter(inputTree))
    #下一个字典
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel


def test(myTree, keyword, featLabels, X_test):
    indexlist = []
    for feat in featLabels:
        index = np.argwhere(keyword == feat).item()
        indexlist.append(index)

    testVec = X_test[:, indexlist]
    pre_labels = []
    for test_sample in testVec:
        pre_label = classify(myTree, featLabels, test_sample)
        pre_labels.append(pre_label)

    return pre_labels
    # cnt = 0
    #
    # for i, test_sample in enumerate(testVec):
    #     pre_label = classify(myTree, featLabels, test_sample)
    #     if pre_label == y_test[i]:
    #         cnt += 1
    # accuracy = cnt / len(y_test)
    # return accuracy


def main():
    #import data
    X_train, X_val, X_test, y_train, y_val, y_test, keyword = loadData()
    dataSet_train = np.hstack((X_train, y_train))

    #val
    # accuracys = []
    # for thresh in [0.1,0.15,0.2,0.3,0.5,0.7]:
    #     featLabels = []
    #     myTree = createTree(dataSet_train, keyword, featLabels, thresh=thresh)
    #     # 测试数据
    #     # testVec：测试数据列表，顺序对应最优特征标签
    #     pre_labels = test(myTree, keyword, featLabels, X_val)
    #     accuracy = np.sum(pre_labels == y_val.reshape(1,-1)[0]) / len(y_val)
    #     accuracys.append(accuracy)
    # print(accuracys)

    #test
    thresh = 0.1
    featLabels = []
    myTree = createTree(dataSet_train, keyword, featLabels, thresh=thresh)
    pre_labels = test(myTree, keyword, featLabels, X_test)
    accuracy = np.sum(pre_labels == y_test.reshape(1,-1)[0]) / len(y_test)
    print(accuracy)


if __name__ =='__main__':
    main()