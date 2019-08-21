# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:24:00 2019

@author: joao_
"""

#%%
import scipy as sp
from scipy import linalg
import numpy as np
from sklearn import datasets

#%%
iris = datasets.load_iris()

X = iris.data

#%%
def mahalanobis(x=None, y=None, cov=None):
    x_minus_mu = x - y
    
    inv_covmat = linalg.inv(cov)
    
    aux = np.dot(x_minus_mu.T, inv_covmat)
    mahal = np.sqrt(np.dot(aux, x_minus_mu))
    
    return mahal

#%%
xx = X[148, :]
yy = X[149, :]

#%% covariance matrix
covmat = np.cov(X.T)
inv_covmat = linalg.inv(covmat)

#%% usando a distancia do scipy
dist1 = sp.spatial.distance.mahalanobis(xx,yy,inv_covmat)

#%% usando a funcao criada
dist2 = mahalanobis(xx, yy, covmat)