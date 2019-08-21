import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from sklearn import datasets

def pearson(x,y):
    medx = x.mean()
    medy = y.mean()
    dpx = x.std()
    dpy = y.std()
    r = 0
    n = x.shape[0]
    for i in range(n):
        r += ((x[i]-medx)/dpx)*((y[i]-medy)/dpy)
    return r


def correlacao(x):
    n,m = x.shape
    r = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            r[i][j] = (1/n) * pearson(x[:,i],x[:,j])
    return r

def mahalanobis(x,y,C):
    return np.sqrt(np.dot(np.dot((x-y).T,np.linalg.inv(C)),(x-y)))

iris = datasets.load_iris()
X = iris.data
riris = correlacao(X)
x1 = X[148,:]
x2 = X[149,:]

dist1 = mahalanobis(x1,x2,riris)
dist2 = scipy.spatial.distance.mahalanobis(x1,x2,np.linalg.inv(riris))

print('Distância da nossa função: ',dist1)
print('Distância do scipy: ',dist2)
