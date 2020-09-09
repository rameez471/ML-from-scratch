import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

X,y = datasets.make_regression(n_samples=100,n_features=4,noise=20,random_state=24)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=24)

def mse(y_pred,y_true):
    return np.mean((y_pred - y_true)**2)

# plt.figure(figsize=(8,6))
# plt.scatter(X_train[:,0],y_train,color='r',marker='o')
# plt.show()

regresoor = LinearRegression(n_iterations=10000)
regresoor.fit(X_train, y_train)
y_pred = regresoor.predict(X_test)

error = mse(y_pred,y_test)
print(error)

y_pred_line = regresoor.predict(X)


if X.shape[1]==1:
    plt.figure(figsize=(8,6))
    plt.scatter(X_train,y_train,color='r')
    plt.scatter(X_test,y_test,color='b')
    plt.plot(X,y_pred_line,color='black',label='Prediction')
    plt.show()