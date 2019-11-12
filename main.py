from sklearn import datasets
from knn import KMeans

iris = datasets.load_iris()
input_dataset = iris.data
target = iris.target

number_of_clusters = 3
k_means = KMeans(number_of_clusters)
knn_result = k_means.fit(input_dataset)
print(knn_result)

eval_matrix = k_means.evaluation_matrix(target)
print(eval_matrix)