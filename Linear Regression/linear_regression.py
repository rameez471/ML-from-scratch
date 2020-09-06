import numpy as np

class LinearRegression:
    
    def __init__(self,n_iterations=1000,learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self,X,y):

        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(n_features)

        for i in range(self.n_iterations):
            y_pred = self.predict(X)

            dw = (1/n_samples) * np.dot(X.T,(y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.learning_rate*dw  
            self.bias = self.bias - self.learning_rate*db


    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

