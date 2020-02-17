import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# leemos los datos

data = np.load('spectra_zoo_reconstructed.npz')
p_el = data['p_el']
p_cs = data['p_cs']

spectra = data['spectra']

for i in range(len(spectra)):
    spectra[i] /= np.sqrt(np.sum(spectra[i]**2))
    

# queremos buscar la galaxia mas eliptica y mas espiral
# con np.argmax vemos cual tiene la mayor proporcion de votos
# las cuales son p_el[32] y p_cs[2016]
  
mu = np.mean(spectra,axis=0)

spectra -= mu

# hacemos la matriz de covarianza

cov = np.cov(spectra.T)


# leemos los autovalores y autovectores de esa matriz

eigVal,eigVec = la.eig(cov)

eigVec = eigVec.T


# las galaxias con las que trabajaremos seran

N=15

eigVec = eigVec[:N]


s_cs = spectra[2016]    # espiral

coef_cs = []

for i in range(N):
    a = np.dot(s_cs,eigVec[i])
    coef_cs.append(a)
    
coef_cs = np.array(coef_cs)

Vcs = np.dot(eigVec.T,coef_cs)


plt.plot(s_cs,label='Real',linewidth=.75,alpha=.5)
plt.plot(Vcs,label='CL ('+str(N)+' componentes)',linewidth=.75,alpha=1,color='r')
plt.xlabel('Longitud de onda')
plt.ylabel('Flujo normalizado')
plt.title('Espiral')
plt.legend(frameon=True,facecolor='w')
plt.show()



s_el = spectra[32]  # eliptica

coef_el = []

for i in range(N):
    a = np.dot(s_el,eigVec[i])
    coef_el.append(a)
    
coef_el = np.array(coef_el)

Vel = np.dot(eigVec.T,coef_el)


plt.plot(s_el,label='Real',linewidth=.75,alpha=.5)
plt.plot(Vel,label='CL ('+str(N)+' componentes)',linewidth=.75,alpha=1,color='r')
plt.xlabel('Longitud de onda')
plt.ylabel('Flujo normalizado')
plt.title('Eliptica')
plt.legend(frameon=True,facecolor='w')
plt.show()