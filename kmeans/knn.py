from scipy.spatial import distance
import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.__centroids = None
    
    def __init_centroids(self, dataset):
        random_centroids = random.sample(list(dataset), self.n_clusters)
        self.__centroids = np.array(random_centroids)

    def fit(self, dataset):
        self.__init_centroids(dataset)

        while True:
            data_clusters = self.__assign_data_clusters(dataset)
            new_centroids = self.__assign_new_centroids(data_clusters)
            if (new_centroids == self.__centroids).all():
                break
            else:
                self.__centroids = new_centroids

            return self

    def __assign_data_clusters(self, dataset):
        data_clusters = [[] for i in range(self.n_clusters)]
        for data in dataset:
            cluster = self.__clusterize_data(data)
            data_clusters[cluster].append(data)
        return data_clusters

    def __clusterize_data(self, data):
        distances = []
        for centroid in self.__centroids:
            dist = distance.euclidean(data, centroid)
            distances.append(dist)
        return np.argmin(distances)

    def __assign_new_centroids(self, data_clusters):
        new_centroids = []
        for data_cluster in data_clusters:
            centroid = np.mean(data_cluster, axis = 0)
            new_centroids.append(centroid)
        return np.array(new_centroids)

    def predict(self, dataset):
        if self.__centroids is None:
            raise Exception("This KMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        clusters = []
        for data in dataset:
            cluster = self.__clusterize_data(data)
            clusters.append(cluster)
        return np.array(clusters)
