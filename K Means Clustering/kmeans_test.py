import numpy as np
import matplotlib.pyplot as plt
from kmeans import *

X = np.array([[0,1],[1,0],[1,2],[2,1],
              [6,6],[5,5],[5,6],[6,5],
              [1,9],[1,10],[1,8],[1,11]])

colors = 10 * ['g','r','c','b','k']

clf = Kmeans(k=2)
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classes:
    color = colors[classification]
    for featureset in clf.classes[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

unknowns = np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4],])

for unknown in unknowns:
   classification = clf.predict(unknown)
   plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)


plt.show()
