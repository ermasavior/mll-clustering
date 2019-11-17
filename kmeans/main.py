from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import sklearn.cluster as sklearn_cluster
import knn
import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target
n_clusters = len(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("Total train dataset: ", len(y_train))
print("Total test dataset: ", len(y_test))

print("\n===========================\n")

print("KNN Clustering from Scratch")
print("> With train and test split")
y_predict = knn.KMeans(n_clusters).fit(X_train).predict(X_test)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y_test, conf_matrix)
print("Accuracy score:", accuracy_score)

print()

print("> Without train and test split")
y_predict = knn.KMeans(n_clusters).fit(X).predict(X)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")

print("KNN Clustering SKLearn")
print("> With train and test split")
y_predict = sklearn_cluster.KMeans(n_clusters).fit(X_train).predict(X_test)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y_test, conf_matrix)
print("Accuracy score:", accuracy_score)

print()

print("> Without train and test split")
y_predict = sklearn_cluster.KMeans(n_clusters).fit(X).predict(X)

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y, y_predict)
print(conf_matrix)

accuracy_score = metrics.cluster_accuracy_score(y_predict, y, conf_matrix)
print("Accuracy score:", accuracy_score)

print("\n===========================\n")
