import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# leemos los datos

data = np.load('spectra_zoo_reconstructed.npz')

spectra = data['spectra']

for i in range(len(spectra)):
    spectra[i] /= np.sqrt(np.sum(spectra[i]**2))
    
    
for i in range(200):
    plt.plot(spectra[i],alpha=0.5,color='black')

mu = np.mean(spectra,axis=0)

plt.plot(mu,color='r',label='Media de todos los espectros')
plt.xlabel('Longitud de onda')
plt.ylabel('Flujo normalizado')
plt.legend(frameon=True,facecolor='w')
plt.show()

spectra -= mu

# hacemos la matriz de covarianza

cov = np.cov(spectra.T)


# leemos los autovalores y autovectores de esa matriz

eigVal,eigVec = la.eig(cov)

eigVec = eigVec.T
# realizamos los graficos de los primeros 4 autovectores

v1 = eigVec[0]
v2 = eigVec[1]
v3 = eigVec[2]
v4 = eigVec[3]

f, axs = plt.subplots(nrows=2,ncols=2,figsize=(15,10))

axs[0,0].plot(v1,label='V1')
axs[0,1].plot(v2,label='V2')
axs[1,0].plot(v3,label='V3')
axs[1,1].plot(v4,label='V4')

axs[0,0].legend(frameon=True,facecolor='w')
axs[0,1].legend(frameon=True,facecolor='w')
axs[1,0].legend(frameon=True,facecolor='w')
axs[1,1].legend(frameon=True,facecolor='w')

axs[0,0].set_ylabel('Flujo normalizado - media')
axs[1,0].set_ylabel('Flujo normalizado - media')
axs[1,0].set_xlabel('Longitud de onda')
axs[1,1].set_xlabel('Longitud de onda')

plt.show()

plt.semilogy()
plt.plot(range(len(eigVal)),eigVal)
plt.show()