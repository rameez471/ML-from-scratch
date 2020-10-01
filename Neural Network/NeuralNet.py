import numpy as np
import matplotlib.pyplot as plt

def activation(z,derivative=False):
    if derivative:
        return activation(z) * (1 - activation(z))
    return 1 / (1+np.exp(-z))

def cost_function(y_true, y_pred):
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_pred - y_true)**2)
    return cost

def cost_function_prime(y_true, y_pred):
    cost_prime = y_pred - y_true
    return cost_prime

class NeuralNetwork(object):

    def __init__(self,size,seed=24):
        self.seed = seed
        np.random.seed(seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i],size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1,len(self.size))]
        self.biases = [np.random.rand(n,1) for n in self.size[1:]]

    def forward(self,inputs):

        a = inputs
        pre_activations = []
        activations = [a]
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,a) + b
            a = activation(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations

    def compute_deltas(self, pre_activations, y_true, y_pred):
        
        delta_L = cost_function_prime(y_true, y_pred) * activation(pre_activations[-1].derivative=True)
        deltas = [0] * (len(self.size)-1)
        deltas[-1] = delta_L
        for l in range(len(deltas)-2, -1, -1):
            delta = np.dot(self.weights[l+1].transpose(),deltas[l+1]*activation(pre_activations[l],derivative=True))
            deltas[l] = delta
        return deltas


    def backpropagate(self, deltas, pre_activations, activations):
        
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            dW_l = np.dot(deltas[l],activations[l-1].transpose())
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1),1))
        return dW,db

    def plot_decision_regions(self,X,y,iterations,train_loss,val_loss,train_acc,val_acc,res=0.01):
        X, y = X.T,y.T
        x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
        xx, yy = np.meshgrid(np.arrange(x_min,x_max,res),
                            np.arrange(y_min,y_max,res))

        Z = self.predict(np.c_[xx.ravel(),yy.ravel()].T)
        Z = Z.reshape(xx.shape)
        plt.contour(xx,yy,Z,alpha=0.5)
        plt.xlim(xx.min(),xx.max())
        plt.ylim(yy.min(),yy.max())
        plt.scatter(X[:,0],X[:,1],c=y.reshape(-1),alpha=0.2)
        message = 'iteration: {} | train loss: {} | val loss: {} | train acc: {} | val acc: {}'.format(iteration,
                                                                                                     train_loss, 
                                                                                                     val_loss, 
                                                                                                     train_acc, 
                                                                                                     val_acc)
        plt.title(message)                                                                                             