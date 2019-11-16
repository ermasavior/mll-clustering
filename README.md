# MLL Clustering From Scratch

Clustering algorithms implementations (K-Means, Agglomerative, and DBSCAN).

## Dependencies
1. Python >=3.6
2. SKLearn
3. SciPy

## Documentations

### KNN
  K-Means clustering.

  ```
  knn.KMeans(n_clusters)
  ```
  **Parameters**
  - **n_clusters:** int. _Number of clusters (predefined)._

  **Methods**
  - `fit(self, input_dataset)`: Compute k-means clustering.
    
    - **Parameters**
      
      - **input_dataset:** array, shape = [n_rows, n_features]. _List of input features._

  - `predict(self, input_dataset)`: Clusterize input with pretrained model.
    
    - **Parameters**
      
      - **input_dataset:** array, shape = [n_rows, n_features]. _List of input features._

    - **Returns**

      - **predicted_clusters:** array, shape = [n_rows]. _Clustering results._


### Metrics: Confusion Matrix
  Generate confusion matrix to evaluate the accuracy of a clustering.

  ``` 
  metrics.cluster_confusion_matrix(n_clusters, predicted_clusters, actual_clusters)
  ```

  **Parameters**
  - **n_clusters:** int. _Number of clusters (predefined)._
  - **predicted_clusters:** array, shape = [n_rows]. _Clustering result._
  - **actual_clusters:** array, shape = [n_rows]. _Ground truth (correct) target values._

  **Returns**
  - **confusion_matrix:** array, shape = [n_clusters , n_clusters]. _Confusion matrix_

## How to Run

Sample usage is demonstrated in `main.py` using [iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris).

1. Run `python main.py`.
2. Program will output the clustering result of iris dataset with the confusion matrix.

## Authors
