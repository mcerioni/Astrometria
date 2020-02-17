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

eigVal_sum = np.cumsum(eigVal)

eigVal_sum /= eigVal_sum[-1]

plt.plot(range(len(eigVal_sum)),eigVal_sum,zorder=1)
plt.xlabel('Cantidad de autovalores ordenados de mayor a menor')
plt.ylabel('Semejanza al espectro observado')
plt.scatter(0,eigVal_sum[0],color='red',zorder=2,label='Primer autovalor. Semejanza de '+str(np.round(eigVal_sum[0],decimals=2)))
plt.hlines(0.95,0,1000,linestyle='dashed',color='red',label='Semejanza de 0.95')
plt.ylim(0,1)
plt.legend(frameon=True,facecolor='w',loc=4)
plt.show()

plt.plot(range(len(eigVal_sum)),eigVal_sum,zorder=1)
plt.xlabel('Cantidad de autovalores ordenados de mayor a menor')
plt.ylabel('Semejanza al espectro observado')
plt.scatter(0,eigVal_sum[0],color='red',zorder=2,label='Primer autovalor. Semejanza de '+str(np.round(eigVal_sum[0],decimals=2)))
plt.hlines(0.95,0,1000,linestyle='dashed',color='red',label='Semejanza de 0.95')
plt.legend(frameon=True,facecolor='w',loc=4)
plt.show()




# np.logspace(3.5,3.9,1000)[n] es la longitud de onda en angstroms que aparece como la longitud de onda n al leer el espectro
