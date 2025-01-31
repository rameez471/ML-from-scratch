import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#ff0000','#00ff00','#0000ff'])
from knn import KNN

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#Load iris dataset 
iris = datasets.load_iris()
X, y = iris.data, iris.target

#Split data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap,edgecolor='k',s=20)
plt.show()

k = 3
clf = KNN(k=k)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = accuracy(y_test, predictions)

print("KNN Classification Score ",score)