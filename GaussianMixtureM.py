# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:05:22 2021

@author: Usuario
"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


xi= np.loadtxt("prueba1_features.txt")

#escalamiento de datos
from sklearn import preprocessing
#normalizamos
xi= preprocessing.normalize(xi,axis=0)

#gaussianizamos
xi=np.log(0.0000000001+xi)#ponemos el 0.0001+ para evitar los 0 porque con el log se nos va a inf
plt.scatter(np.arange(xi.shape[0]), xi[:, 0])

#aplicamos reducción de dimensionalidad
pca= PCA(n_components=1)
pca.fit(xi)
xi=pca.transform(xi)


#creamos el gaussian mixture model
gmm= GaussianMixture(n_components=2, covariance_type="full", n_init=3)
gmm.fit(xi)

predicciones=gmm.predict(xi)

for i, pred in enumerate(predicciones):
    print("muestra", i, "se encuentra en el cluster:", pred)
    
    

colores = ['red','blue']

colores_cluster = [colores[predicciones[i]] for i in range(len(xi))]
 
plt.show()  

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(xi, predicciones)
print("fiabilidad",db_index)
plt.show()
plt.scatter(np.arange(xi.shape[0]), xi[:, 0],c=colores_cluster)