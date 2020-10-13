#!/usr/bin/python
# -*- coding:utf-8 -*-
# 1.KNN on MNIST, use 100, 300, 1000, 3000, 10000, 30000, 60000 training samples and compare
# 2.different k and compare
# 3.three different distance metric and compare(L0, L1, L2）
# Reference: https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_mnist.html

import torch
import csv
import numpy as np


def main():
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    #import data
    with open('mnist_test.csv', newline='') as csv_file2:
        test_data_lines = csv.reader(csv_file2)
        test_dataset = list(test_data_lines)
        # Converting list into matrix and changing Datatype into int
        test_matrix = np.array(test_dataset).astype("float32")
        x_test = test_matrix[:, 1:785]/255 # normalization
        y_test = test_matrix[:, 0]
        x_test = torch.tensor(x_test, dtype=torch.float).cuda(3)
        y_test = torch.tensor(y_test, dtype=torch.int).cuda(3)

    with open('mnist_train.csv', newline='') as csv_file1:
        train_data_lines = csv.reader(csv_file1)
        train_dataset = list(train_data_lines)

        train_matrix = np.array(train_dataset).astype("float32")
        x_train_all = train_matrix[:,1:785]/255 # normalization
        y_train_all = train_matrix[:,0]

    for num in [100, 300, 1000, 3000, 10000, 30000, 60000]:
        x_train = x_train_all[0:num]
        x_train = torch.tensor(x_train, dtype=torch.float).cuda(3)
        y_train = y_train_all[0:num]
        y_train = torch.tensor(y_train, dtype=torch.int).cuda(3)
        print(num)

        for k in [1,3,5,7,9]:
            X_i = x_test[:, None, :] # (10000, 1, 784) test set
            X_j = x_train[None, :, :]  # (1, 60000, 784) train set
            D_ij = []
            for step in range(int(num/100)): # 100 samples at a time to avoid cuda out of memory
                D_ij_temp = ((X_i - X_j[:,100*step:100*(step+1),:]) ** 2).sum(-1) # (10000, N_train) symbolic matrix of squared L2 distances
                # D_ij_temp = (abs(X_i - X_j[:, 100 * step:100 * (step + 1), :])).sum(-1) # (10000, N_train) symbolic matrix of squared L1 distances
                # D_ij_temp = ((X_i - X_j[:, 100 * step:100 * (step + 1), :]) !=0).sum(-1) # (10000, N_train) symbolic matrix of squared L0 distances

                D_ij.append(D_ij_temp)
            D_ij = torch.cat(D_ij, dim=1) # torch append

            _, ind_knn = torch.sort(D_ij, dim=1)  # Sorting distances with ascending order
            lab_knn = y_train[ind_knn[:, 0:k]]  # (N_test, K) array of integers in [0,9]
            y_knn, _ = torch.mode(lab_knn, dim=1)  # Compute the most likely label

            if use_cuda: torch.cuda.synchronize()

            accuracy = 1-(y_knn != y_test).float().mean().item()
            print("{}-NN dis-L1 on the full MNIST dataset: test accuracy = {:.2f}%.".format(k, accuracy * 100))


if __name__ == "__main__":
    main()