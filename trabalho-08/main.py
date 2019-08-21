# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:50:36 2019

@author: joao_
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
def gaussian(x, mu, sigma):
    return  (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(1/2) * np.power((x-mu)/sigma, 2))

def gaussian_2D(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return (1/(2 * np.pi * sigma_x * sigma_y)) * np.exp(-(1/2)*((np.power((x-mu_x)/sigma_x, 2)) + (np.power((y-mu_y)/sigma_y, 2))))

def gaussian_2D_cor(x,y,mu,sigma,p):
    z = np.power(((x-mu[0])/sigma[0]), 2) - 2*p*((x-mu[0])/sigma[0])*((y-mu[1])/sigma[1]) + np.power(((y-mu[1])/sigma[1]) , 2)
    return (1/(2*np.pi*sigma[0]*sigma[1]*np.sqrt(1-np.power(p, 2)))) * np.exp(-z/(2*(1-np.power(p, 2))))

#%% gaussiana
mu, sigma = 1, 1

x = np.arange(-15,15,0.1)

fig = plt.figure()
plt.plot(x, gaussian(x, mu, sigma))
plt.show()

#%% gaussiana 2d
mediax = 0
mediay = 0
dpx = 1 
dpy = 1

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
xx,yy = np.meshgrid(x,y)

z = gaussian_2D(xx,yy,mediax,mediay,dpx,dpy)

fig = plt.figure()
axes3D = Axes3D(fig)
axes3D.plot_surface(xx,yy,z)
plt.show()

#%% gaussiana 2d com corelacao
media = [0, 0]
dp = [1, 1]

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
xx,yy = np.meshgrid(x,y)


z = gaussian_2D_cor(xx, yy, media, dp, 0.1)

fig = plt.figure()
axes3D = Axes3D(fig)
axes3D.plot_surface(xx,yy,z,cmap='viridis')
plt.show()