import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from naive_bayes import NaiveBayes

X,y = datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=24)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

def accuracy(y_true,y_pred):
    accuracy = np.sum(y_pred == y_true)/len(y_true)
    return accuracy

nb = NaiveBayes()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)

print('Naive Bayes Accuracy: ',accuracy(y_test, predictions))