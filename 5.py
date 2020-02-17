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

N=10
Nespectros=4000

eigVec = eigVec[:N]

comp_total = []

for i in range(N):
    comp_total.append([])

for j in range(Nespectros):
    
    s = spectra[j]    
    
    coef = np.zeros(N)
    
    for i in range(N):
        a = np.dot(s,eigVec[i])
        coef[i] = a
    
    Vcs = np.dot(eigVec.T,coef)

    for k in range(len(coef)):
        comp_total[k].append(coef[k])

for i in comp_total:
    i = np.array(i)    
    
comp_total = np.array(comp_total)
f, axs = plt.subplots(nrows=N,ncols=N,figsize=[N*3,N*3])

ids_el = p_el > 0.7
ids_cs = p_cs > 0.7

ids_el = ids_el[:Nespectros]
ids_cs = ids_cs[:Nespectros]


for j in range(len(comp_total)):
    for k in range(len(comp_total)):
        if k != j:
            
            axs[j,k].scatter(comp_total[j][ids_el],comp_total[k][ids_el],color='coral',label='Elipticas',alpha=0.5,s=1.5,zorder=2)
            axs[j,k].scatter(comp_total[j][ids_cs],comp_total[k][ids_cs],color='cadetblue',label='Espirales',alpha=0.5,s=1.5,zorder=1)
            
        if k==0:
            axs[j,0].set_ylabel('V '+str(j+1))
        if j==(len(comp_total)-1):
            axs[len(comp_total)-1,k].set_xlabel('V '+str(k+1))

for j in range(len(comp_total)):
    axs[j,j].hist(comp_total[j][ids_el],color='coral',alpha=0.5)
    axs[j,j].hist(comp_total[j][ids_cs],color='cadetblue',alpha=0.5)
    

plt.show()