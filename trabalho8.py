import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_1D(x,media,dp):
    return (1/(dp*np.sqrt(2*np.pi))) * np.exp((-1/2)*np.power((x-media)/dp,2))

def gaussian_2D_identidade(x,y,mediax,mediay):
    return (1/(2*np.pi))*np.exp(-0.5*((x-mediax)**2+(y-mediay)**2))

def gaussian_2D(x,C,med):
    A = (1/(2*np.pi)) * 1/np.sqrt(np.linalg.det(C))
    aux = x - med
    B = np.exp(-0.5*(np.dot(np.dot(aux.T,np.linalg.inv(C)),aux)))
    return A*B

#%%
# 1D
media = 1
dp = 5
x = np.arange(-15,15,0.1)
y = gaussian_1D(x,media,dp)

plt.figure()
plt.plot(x,y)
plt.grid()
plt.title('Gaussiana 1-D (Média = {} Desvio Padrão = {})'.format(media,dp))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#%%
# 2D correlacao identidade
mediax = 0
mediay = 0
dpx = 1
dpy = 1
x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
xx,yy = np.meshgrid(x,y)
z = gaussian_2D_identidade(xx,yy,mediax,mediay)
fig = plt.figure()
axes3D = Axes3D(fig)
axes3D.plot_surface(xx,yy,z)
plt.show()

#%%
# 2D correlacao

mediax = 0
mediay = 0
dpx = 5
dpy = 1

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
xx,yy = np.meshgrid(x,y)

C = np.zeros([2,2])
C[0,0] = dpx**2 
C[0,1] = dpx * dpy
C[1,0] = dpx*dpy
C[1,1] = dpy**2

z = np.zeros(xx.shape) 
aux = np.zeros([2,1])
media = np.zeros([2,1])

media[0][0] = mediax
media[1][0] = mediay

for i in range(xx.shape[0]):
    for j in range(yy.shape[0]):
        aux[0][0] = xx[i][j]
        aux[1][0] = yy[i][j]
        z[i][j] = gaussian_2D(aux,C,media)



