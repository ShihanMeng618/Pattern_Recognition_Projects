#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from math import log
import scipy.io as scio
import pandas as pd
import operator
from sklearn.model_selection import train_test_split


def loadData():
     data = scio.loadmat("Sogou_data/Sogou_webpage.mat")
     word_mat = data['wordMat'] # 14400*1200 doc_num * keyword_num
     doc_label = data['doclabel']# 14400*1 每篇文档对应的标签
     data_frame = pd.read_excel('Sogou_data/keyword.xls')
     keyword = data_frame.values[:,1] # 1200个关键词
     # 3:1:1 train:validation:test
     X_train, X_test, y_train, y_test = train_test_split(word_mat, doc_label, test_size=0.2)
     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
     return X_train, X_val,X_test, y_train, y_val, y_test, keyword


def GenerateTree(dataSet,labels,featLabels,thresh):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])/len(classList) > thresh:
        return classList[0]
    if len(dataSet[0])==1:
        major, _ = majorCnt(classList)
        return major
    bestFeat=SelectFeature(dataSet)
    bestFeatLabel=labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree={bestFeatLabel.tolist():{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVls = set(featValues)
    for value in uniqueVls:
        myTree[bestFeatLabel][value] = GenerateTree(SplitNode(dataSet, bestFeat, value),
                                                  labels, featLabels)
    return myTree


def SplitNode(dataSet, axis, value): #对当前节点进行分支
    # 创建返回的数据集列表
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet: #每一行
        if featVec[axis] == value:
            # 去掉axis特征
            reduceFeatVec = featVec[:axis].tolist()
            # 将符合条件的添加到返回的数据集
            reduceFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reduceFeatVec)
    # 返回划分后的数据集
    return retDataSet


def SelectFeature(dataSet): #对当前节点选择待分特征
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = Impurity(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = SplitNode(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * Impurity((subDataSet))
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def Impurity(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt  
    
    
def majorCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
            classCount[vote]+=1
        #降序
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        mis_impurity = 1 - sortedClassCount[0][0] / np.sum(classCount.items()) #采用错分不纯度
        return sortedClassCount[0][0], mis_impurity


# 使用生成的树GenerateTree，对样本XToBePredicted进行预测
def Decision(inputTree,featLabels,testVec):
    firstStr=next(iter(inputTree))
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=Decision(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel


# def Prune(GenerateTree, CrossValidationDataset):#对已经生成好并停止分支的树进行剪枝：
    # 考虑所有相邻的叶子节点，如果将他们消去可以增加验证集上的正确率，则减去两叶子节点，将他们的共同祖先作为新的叶子节点


def main():
    #import data
    X_train, X_val, X_test, y_train, y_val, y_test, propertyName = loadData()
    featLabels = []
    myTree = GenerateTree(X_train, y_train, featLabels)

    # 测试数据
    pre_label = Decision(myTree, featLabels, X_test)

    print(accuracy)

if __name__ =='__main__':
    main()
