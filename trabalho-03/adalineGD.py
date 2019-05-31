# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:42:06 2019

@author: joao_
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#%% Adaline class
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input   = self.net_input(X)
            output      = self.activation(net_input)
            errors      = (y-output)
            self.w_[1:] += self.eta * X.T.dot(net_input)
            self.w_[0]  += self.eta * errors.sum()
            cost = (errors**2).sum() /2.0
            self.cost_.append(cost)
            
        return self
    
    def fit_plot(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input   = self.net_input(X)
            output      = self.activation(net_input)
            errors      = (y-output)
            self.w_[1:] += self.eta * X.T.dot(net_input)
            self.w_[0]  += self.eta * errors.sum()
            cost = (errors**2).sum() /2.0
            self.cost_.append(cost)
            
            self.plot_decision_regions(X, y)
            
            plt.xlabel('sepal length [cm]')
            plt.ylabel('petal length [cm]')
            plt.legend(loc='upper left')
            plt.title('n_iter='+str(i+1))
            plt.show()
        
        return self
    
    def plot_decision_regions(self, X, y, resolution=0.02):
        
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
        
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        alpha=0.8, 
                        c=colors[idx],
                        marker=markers[idx], 
                        label=cl, 
                        edgecolor='black')
        return
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
#%% Data set
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

# select seteosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.xlabel('sepal length [cm]')
plt.xlabel('sepal length [cm]')
plt.xlabel('sepal length [cm]')

plt.show()

#%% Trainning the AdalineGD Model
ada = AdalineGD(eta=0.001, n_iter=10)

ada.fit_plot(X, y)

