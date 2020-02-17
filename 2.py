import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# leemos los datos

data = np.load('spectra_zoo_reconstructed.npz')

spectra = data['spectra']


# hacemos la matriz de covarianza

cov = np.cov(spectra.T)


# leemos los autovalores y autovectores de esa matriz

eigVal,eigVec = la.eig(cov)


# realizamos los graficos de los primeros 4 autovectores

v1 = eigVec[0]
v2 = eigVec[1]
v3 = eigVec[2]
v4 = eigVec[3]

f, axs = plt.subplots(nrows=2,ncols=2)

axs[0,0].plot(v1,label='V1')
axs[0,1].plot(v2,label='V2')
axs[1,0].plot(v3,label='V3')
axs[1,1].plot(v4,label='V4')
#plt.legend(frameon=True,facecolor='w')

plt.show()


# np.logspace(3.5,3.9,1000)
