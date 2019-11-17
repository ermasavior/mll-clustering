from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import dbscan as db

# import dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into test and train, but sort ordinal first to reduce mislabel
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
np.size(X_train,0)
X_temp = np.concatenate((X_train, np.array([y_train]).T), axis=1)
X_temp = X_temp[X_temp[:, np.size(X,1)-1].argsort()]
X_train = np.array([i[:-1] for i in X_temp.tolist()])
y_train = np.array([i[-1] for i in X_temp.tolist()]).astype(int)

# do algorithm
print("======== minpts : 2 && eps : 0.3 ========")
print("WITH SPLIT ", len(y_train), ":", len(y_test))
print("\nDBSCAN Clustering from Scratch")
model1_mn = db.DBSCAN_mn(0.3,2)
model1_mn.fit(X_train)
y_predict = model1_mn.predict(X_test)
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_predict))
print("Accuracy Score :")
print(accuracy_score(y_test, y_predict))

print("______________________________")
print("\nWITHOUT SPLIT")
print("\nDBSCAN Clustering from scratch")
model1 = db.DBSCAN_mn(0.3, 2)
model1.fit(X)
pred1_mn = model1.predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred1_mn))
print("Accuracy Score :")
print(accuracy_score(y, pred1_mn))

print("\nDBSCAN Clustering from sklearn")
pred1_sk = DBSCAN(eps=0.3, min_samples=2).fit_predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred1_sk))
print("Accuracy Score :")
print(accuracy_score(y, pred1_sk))

print("\n\n")

print("======== minpts : 2 && eps : 0.7 ========")
print("WITH SPLIT ", len(y_train), ":", len(y_test))
print("\nDBSCAN Clustering from Scratch")
model2_mn = db.DBSCAN_mn(0.7,2)
model2_mn.fit(X_train)
y_predict2 = model2_mn.predict(X_test)
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_predict2))
print("Accuracy Score :")
print(accuracy_score(y_test, y_predict2))

print("______________________________")
print("\nWITHOUT SPLIT")
print("\nDBSCAN Clustering from scratch")
model2 = db.DBSCAN_mn(0.7, 2)
model2.fit(X)
pred2_mn = model2.predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred2_mn))
print("Accuracy Score :")
print(accuracy_score(y, pred2_mn))

print("\nDBSCAN Clustering from sklearn")
pred2_sk = DBSCAN(eps=0.7, min_samples=2).fit_predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred2_sk))
print("Accuracy Score :")
print(accuracy_score(y, pred2_sk))

print("\n\n")

print("======== minpts : 2 && eps : 3 ========")
print("WITH SPLIT ", len(y_train), ":", len(y_test))
print("\nDBSCAN Clustering from Scratch")
model3_mn = db.DBSCAN_mn(3,2)
model3_mn.fit(X_train)
y_predict3 = model3_mn.predict(X_test)
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_predict3))
print("Accuracy Score :")
print(accuracy_score(y_test, y_predict3))

print("______________________________")
print("\nWITHOUT SPLIT")
print("\nDBSCAN Clustering from scratch")
model3 = db.DBSCAN_mn(3, 2)
model3.fit(X)
pred3_mn = model3.predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred3_mn))
print("Accuracy Score :")
print(accuracy_score(y, pred3_mn))

print("\nDBSCAN Clustering from sklearn")
pred3_sk = DBSCAN(eps=3, min_samples=2).fit_predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred3_sk))
print("Accuracy Score :")
print(accuracy_score(y, pred3_sk))

print("\n\n")

print("======== minpts : 20 && eps : 0.75 ========")
print("WITH SPLIT ", len(y_train), ":", len(y_test))
print("\nDBSCAN Clustering from Scratch")
model4_mn = db.DBSCAN_mn(0.75,20)
model4_mn.fit(X_train)
y_predict4 = model4_mn.predict(X_test)
print("Confusion Matrix :")
print(confusion_matrix(y_test, y_predict4))
print("Accuracy Score :")
print(accuracy_score(y_test, y_predict4))

print("______________________________")
print("\nWITHOUT SPLIT")
print("\nDBSCAN Clustering from scratch")
model4 = db.DBSCAN_mn(0.75, 20)
model4.fit(X)
pred4_mn = model4.predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred4_mn))
print("Accuracy Score :")
print(accuracy_score(y, pred4_mn))

print("\nDBSCAN Clustering from sklearn")
pred4_sk = DBSCAN(eps=0.75, min_samples=20).fit_predict(X)
print("Confusion Matrix :")
print(confusion_matrix(y, pred4_sk))
print("Accuracy Score :")
print(accuracy_score(y, pred4_sk))

print("\n\n")
