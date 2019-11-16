import numpy as np
from sklearn.metrics import accuracy_score

def cluster_confusion_matrix(n_clusters, predicted_clusters, actual_labels):
    evaluation_matrix = [np.zeros(n_clusters) for _ in range(n_clusters)]

    for idx in range(len(predicted_clusters)):
        actual_idx = actual_labels[idx]
        predicted_idx = predicted_clusters[idx]
        evaluation_matrix[predicted_idx][actual_idx] += 1

    return np.array(evaluation_matrix)


def cluster_accuracy_score(predicted_clusters, actual_labels, confusion_matrix):
    transformed_clusters = []

    for cluster in predicted_clusters:
        aligned_cluster = np.argmax(confusion_matrix[cluster])
        transformed_clusters.append(aligned_cluster)
    transformed_clusters = np.array(transformed_clusters)

    return accuracy_score(transformed_clusters, actual_labels)