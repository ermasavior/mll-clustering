import numpy as np
from sklearn.metrics import accuracy_score

def cluster_accuracy_score(predicted_clusters, actual_labels, confusion_matrix):
    transformed_clusters = []
    confusion_matrix = confusion_matrix.T

    for cluster in predicted_clusters:
        aligned_cluster = np.argmax(confusion_matrix[cluster])
        transformed_clusters.append(aligned_cluster)
    transformed_clusters = np.array(transformed_clusters)

    return accuracy_score(actual_labels, transformed_clusters)