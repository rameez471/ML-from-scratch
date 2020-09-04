import numpy as np
from collections import Counter

def euclidean_distance(x,y):
    return np.sqrt(sum((x-y)**2))

class KNN:

    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,X):
        # Calculate distance between x and all examples in the training set
        distances = [euclidean_distance(X,x_train) for x_train in self.X_train]
        # Sort y distance and return indices of first K Neighbors
        k_index = np.argsort(distances)[:self.k]
        # Extract the labels of k nearest neighbors from training set
        k_nearest_labels = [self.y_train[i] for i in k_index]
        # return the most common label
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label        
