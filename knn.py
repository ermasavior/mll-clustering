from scipy.spatial import distance
import numpy as np
import random

class KNN:
    def __init__(self, input_dataset, number_of_clusters = 3):
        self.input_dataset = input_dataset
        self.number_of_clusters = number_of_clusters

    def kmeans(self):
        centroids = self.init_centroids()

        while True:
            self.knn_result = []
            new_clusters = [[] for i in range(self.number_of_clusters)]

            for data in self.input_dataset:
                cluster = self.clusterize_data(data, centroids)
                new_clusters[cluster].append(data)
                self.knn_result.append(cluster)

            new_centroids = []
            for cluster in new_clusters:
                new_centroid = np.mean(cluster, axis = 0)
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            if (new_centroids == centroids).all():
                break
            else:
                centroids = new_centroids

        return self.knn_result
    
    def init_centroids(self):
        return np.array(random.sample(list(self.input_dataset), self.number_of_clusters))

    def clusterize_data(self, data, centroids):
        distances = []
        for centroid in centroids:
            dist = distance.euclidean(data, centroid)
            distances.append(dist)
        return np.argmin(distances)
    
    def evaluation_matrix(self, number_of_row_per_label):
        evaluation_matrix = [np.zeros(self.number_of_clusters) for _ in range(self.number_of_clusters)]
        actual_label = -1
        for idx in range(len(self.knn_result)):
            if idx % number_of_row_per_label == 0: actual_label += 1
            cluster_idx = self.knn_result[idx]
            evaluation_matrix[cluster_idx][actual_label] += 1
        return evaluation_matrix

from sklearn import datasets
input_dataset = datasets.load_iris().data
knn_result = KNN(input_dataset).kmeans()
print(knn_result)