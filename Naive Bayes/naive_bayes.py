import numpy as np  

class NaiveBayes:

    def fit(self,X,y):

        n_samples,n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        ## calculate mean , variance and prior for each class

        self._mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self._var = np.zeros((n_classes,n_features),dtype=np.float64)
        self._prior = np.zeros((n_classes),dtype=np.float64)

        for idx,c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0]/float(n_samples)

    def predict(self,X):
