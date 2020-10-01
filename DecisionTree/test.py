from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decisionTree import DecisionTreeClassifier
from pprint import pprint
import numpy as np

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

clf = DecisionTreeClassifier(max_depth=7, feature_names=iris.feature_names)

m = clf.fit(X_train,y_train)

pprint(m)

predictions = clf.predict(X_test)

print('Test accuracy:',accuracy(y_test, predictions))