import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Kmeans:

    def __init__(self, k=3, tolerance=0.0001,max_iter=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classes = {}

            for i in range(self.k):
                self.classes[i] = []

            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification],axis=0)

            isOptimal = True

            for centroid in self.centroids:
                orignal_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - orignal_centroid)/orignal_centroid*100.0) > self.tolerance:
                    isOptimal = False

            if isOptimal:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


