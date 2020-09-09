import numpy as np

class LogisticRegression:

    def __init__(self,learning_rate=0.001,n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):

            y_pred = self.sigmoid(np.dot(X,self.weights) + self.bias)

            dw = (1/n_samples) * np.dot(X.T,(y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,X):
        y_pred = self.sigmoid(np.dot(X,self.weights) + self.bias)
        predicted_classes = [1 if i>0.5 else 0 for i in y_pred]
        return np.array(predicted_classes)

    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))