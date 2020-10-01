import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

    def train(self,X,y,batch_size,epochs,learning_rate,validation_split=0.2,plot=False):

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=validation_split)
        X_train,X_test,y_train,y_test = X_train.T,X_test.T,y_train.T,y_test.T

        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []

        for i in range(epochs):
            if X_train.shape[1] % batch_size == 0:
                n_batches = int(X_train.shape[1] / batch_size)
            else:
                n_batches = int(X_train.shape[1] / batch_size)-1
            
            X_train, y_train = shuffle(X_train.T, y_train.T)
            X_train, y_train = X_train.T, y_train.T

            batches_X = [X_train[:,batch_size*i:batch_size*(i+1)] for i in range(0,n_batches)]
            batches_y = [y_train[:,batch_size*i:batch_size*(i+1)] for i in range(0,n_batches)]

            train_losses = []
            train_accuracies = []

            test_losses = []
            train_accuracies = []

            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases]
            
            for batch_x, batch_y in zip(batches_X,batch_y):
                batch_y_pred,pre_activations,activations = self.forward(batch_x)
                deltas = self.compute_deltas(pre_activations,batch_y,batch_y_pred)
                dW, db = self.backpropagate(deltas, pre_activations, activations)

                for i,(dw_i,db_i) in enumerate(zip(dW,db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)

                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = accuracy_score(batch_y.T, batch_y_train_pred.T)
                train_accuracies.append(train_accuracy)

                batch_y_test_pred = self.predict(X_test)

                test_loss = cost_function(y_test, y_test_pred)
                test_losses.append(test_loss)
                test_accuracy = accuracy_score(y_test.T,batch_y_pred.T)
                test_accuracies.append(test_accuracy)
            
            for i,(dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))
            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))

        
        self.plot_decision_regions(X, y,epochs,
                                        np.round(np.mean(train_losses),4),
                                        np.round(np.mean(test_losses),4),
                                        np.round(np.mean(train_accuracies),4),
                                        np.round(np.mean(test_accuracies),4))

        history = {
            'epochs':epochs,
            'train_loss':history_train_losses,
            'train_acc':history_train_accuracies,
            'test_loss':history_test_losses,
            'test_acc':history_test_accuracies
        }

    def predict(self,a):

        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,a) + b
            a = activation(z)
        predictions = ( a > 0.5).astype(int)

        return predictions

