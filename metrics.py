import numpy as np

def cluster_confusion_matrix(n_clusters, predicted_clusters, actual_clusters):
    evaluation_matrix = [np.zeros(n_clusters) for _ in range(n_clusters)]

    for idx in range(len(predicted_clusters)):
        actual_cluster_idx = actual_clusters[idx]
        predicted_cluster_idx = predicted_clusters[idx]
        evaluation_matrix[predicted_cluster_idx][actual_cluster_idx] += 1

    return np.array(evaluation_matrix)
