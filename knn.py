from scipy.spatial import distance
import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def __init_centroids(self, input_dataset):
        random_centroids = random.sample(list(input_dataset), self.n_clusters)
        return np.array(random_centroids)

    def fit(self, input_dataset):
        self.__centroids = self.__init_centroids(input_dataset)

        while True:
            knn_result = []
            new_clusters = [[] for i in range(self.n_clusters)]

            for data in input_dataset:
                cluster = self.__clusterize_data(data)
                new_clusters[cluster].append(data)
                knn_result.append(cluster)

            new_centroids = self.__assign_new_centroids(new_clusters)
            if (new_centroids == self.__centroids).all():
                break
            else:
                self.__centroids = new_centroids

        return knn_result

    def __clusterize_data(self, data):
        distances = []
        for centroid in self.__centroids:
            dist = distance.euclidean(data, centroid)
            distances.append(dist)
        return np.argmin(distances)
    
    def __assign_new_centroids(self, new_clusters):
        new_centroids = []
        for cluster in new_clusters:
            centroid = np.mean(cluster, axis = 0)
            new_centroids.append(centroid)
        return np.array(new_centroids)
