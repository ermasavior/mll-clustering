from sklearn import datasets
from knn import KMeans
from metrics import cluster_confusion_matrix

iris = datasets.load_iris()
input_dataset = iris.data
target = iris.target
n_clusters = len(iris.target_names)

print("KNN Clustering Result")
k_means = KMeans(n_clusters)
knn_result = k_means.fit(input_dataset)
print(knn_result)

confusion_matrix = cluster_confusion_matrix(n_clusters, target, knn_result)
print("Confusion Matrix:")
print(confusion_matrix)
