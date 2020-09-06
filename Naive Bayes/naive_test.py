import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from naive_bayes import NaiveBayes

X,y = datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=24)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

nb = NaiveBayes()
nb.fit(X_train,y_train)