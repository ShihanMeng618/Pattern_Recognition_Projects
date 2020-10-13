#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import csv
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from time import time
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def main():
    # import data from MNIST
    with open('mnist_train.csv', newline='') as csv_file1:
        train_data_lines = csv.reader(csv_file1)
        train_dataset = list(train_data_lines)
        train_matrix = np.array(train_dataset).astype("float32")
        x_train = train_matrix[:, 1:785] / 255  # normalization 60000*784
        y_train = train_matrix[:, 0]


    # #K-means
    # for samples in [10,100,1000,10000,60000]:
    #     start = time()
    #     Kmeans_model = KMeans(n_clusters = n)
    #     Kmeans_model.fit(x_train[0:samples,:])
    #     estimate_labels = Kmeans_model.labels_
    #     end = time()
    #     print(end-start)
    #     result = metrics.normalized_mutual_info_score(y_train[0:samples], estimate_labels)
    #     print(result)

    # 4.2 & 4.3
    # samples = 10000
    # Jes = []
    # results = []
    # times = []
    # for n in range(2,16,2):
    #     start = time()
    #     Kmeans_model = KMeans(n_clusters=n)
    #     Kmeans_model.fit(x_train[0:samples,:])
    #     estimate_labels = Kmeans_model.labels_
    #     end = time()
    #     Je = Kmeans_model.inertia_
    #     Jes.append(Je)
    #     result = metrics.normalized_mutual_info_score(y_train[0:samples], estimate_labels)
    #     results.append(result)
    #     times.append(end-start)
    # print(Jes)
    # print(results)
    # print(times)


    # hierarchical clustering
    # for samples in [5000]:#[10, 100, 1000, 5000,10000]:
    #     start = time()
    #     H_model = AgglomerativeClustering(n_clusters=n)
    #     H_model.fit(x_train[0:samples, :])
    #     estimate_labels = H_model.labels_
    #     end = time()
    #     print(end - start)
    #     result = metrics.normalized_mutual_info_score(y_train[0:samples], estimate_labels)
    #     print(result)

    # 4.2
    # samples = 1000
    # linkage = ['ward','complete', 'average', 'single']
    # results = []
    # for i in range(4):
    #     H_model = AgglomerativeClustering(n_clusters=n, linkage = linkage[i])
    #     H_model.fit(x_train[0:samples, :])
    #     estimate_labels = H_model.labels_
    #     result = metrics.normalized_mutual_info_score(y_train[0:samples], estimate_labels)
    #     results.append(result)
    # print(results)

    # plot Hierarchical Clustering Dendrogram
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(x_train[0:10000,:])
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=5)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.ylabel('distance')
    plt.show()


    #spectral clustering
    # for samples in [10, 50, 100, 500, 1000]:
    #     start = time()
    #     S_model = SpectralClustering(n_clusters=n)
    #     S_model.fit(x_train[0:samples, :])
    #     estimate_labels = S_model.labels_
    #     end = time()
    #     print(end - start)
    #     result = metrics.normalized_mutual_info_score(y_train[0:samples], estimate_labels)
    #     print(result)

    # 4.2
    # samples = 100
    # print(y_train[0:samples])
    # n_neighbors = [5,10,20,30]
    # for i in range(4):
    #     start = time()
    #     S_model = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', n_neighbors=n_neighbors[i])
    #     S_model.fit(x_train[0:samples, :])
    #     estimate_labels = S_model.labels_
    #     end = time()
    #     print(end - start)
    #     result = metrics.normalized_mutual_info_score(y_train[0:samples], estimate_labels)
    #     print(result)


if __name__ == '__main__':
    main()