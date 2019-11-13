from sklearn import datasets
from knn import KMeans
from metrics import cluster_confusion_matrix

iris = datasets.load_iris()
input_dataset = iris.data
target = iris.target

number_of_clusters = 3
k_means = KMeans(number_of_clusters)
knn_result = k_means.fit(input_dataset)
print("KNN")
print(knn_result)

confusion_matrix = cluster_confusion_matrix(number_of_clusters, target, knn_result)
print("Confusion Matrix:")
print(confusion_matrix)