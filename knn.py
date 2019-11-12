from scipy.spatial import distance
import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters = 3):
        self.n_clusters = n_clusters

    def fit(self, input_dataset):
        self.input_dataset = input_dataset
        centroids = np.array(self.__init_centroid())

        while True:
            self.knn_result = []
            new_clusters = [[] for i in range(self.n_clusters)]

            for data in self.input_dataset:
                cluster = self.__clusterize_data(data, centroids)
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
    
    def __init_centroid(self):
        return random.sample(list(self.input_dataset), self.n_clusters)

    def __clusterize_data(self, data, centroids):
        distances = []
        for centroid in centroids:
            dist = distance.euclidean(data, centroid)
            distances.append(dist)
        return np.argmin(distances)
    
    def evaluation_matrix(self, actual_clusters):
        evaluation_matrix = [np.zeros(self.n_clusters) for _ in range(self.n_clusters)]
        for idx in range(len(self.knn_result)):
            actual_cluster = actual_clusters[idx]
            predicted_cluster = self.knn_result[idx]
            evaluation_matrix[predicted_cluster][actual_cluster] += 1
        return evaluation_matrix
