import numpy as np
import pandas as pd
import sklearn.cluster as sklearn_cluster
from sklearn import datasets
from sklearn.metrics import confusion_matrix 
from agglomerative_clustering import AgglomerativeClustering
import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target
n_clusters = len(iris.target_names)

print("\n===========================\n")

print("Agglomerative Clustering (Single) from Scratch")
y_predict = AgglomerativeClustering(pd.DataFrame(X), n_clusters, 'single').fit_predict()
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("Agglomerative Clustering (Single) SKLearn")
y_predict = sklearn_cluster.AgglomerativeClustering(linkage='single').fit_predict(X)
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("Agglomerative Clustering (Complete) from Scratch")
y_predict = AgglomerativeClustering(pd.DataFrame(X), n_clusters, 'complete').fit_predict()
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("Agglomerative Clustering (Complete) SKLearn")
y_predict = sklearn_cluster.AgglomerativeClustering(linkage='complete').fit_predict(X)
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("Agglomerative Clustering (Average) from Scratch")
y_predict = AgglomerativeClustering(pd.DataFrame(X), n_clusters, 'average').fit_predict()
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("Agglomerative Clustering (Average) SKLearn")
y_predict = sklearn_cluster.AgglomerativeClustering(linkage='average').fit_predict(X)
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("Agglomerative Clustering (Average-Group) from Scratch")
y_predict = AgglomerativeClustering(pd.DataFrame(X), n_clusters, 'average-group').fit_predict()
print(y_predict)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)