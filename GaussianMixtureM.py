# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:05:22 2021

@author: Usuario
"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

xi= np.loadtxt("prueba1_features.txt")

#escalamiento de datos
from sklearn import preprocessing
xi= preprocessing.normalize(xi,axis=0)

#creamos el gaussian mixture model
gmm= GaussianMixture(n_components=2, covariance_type="full", n_init=3)
gmm.fit(xi)

predicciones=gmm.predict(xi)

for i, pred in enumerate(predicciones):
    print("muestra", i, "se encuentra en el cluster:", pred)
    
    
from sklearn.decomposition import PCA
modelo_pca = PCA(n_components = 2, svd_solver='full')
modelo_pca.fit(xi)
pca = modelo_pca.transform(xi)
colores = ['red','blue']

colores_cluster = [colores[predicciones[i]] for i in range(len(pca))]
 
plt.show()  

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(xi, predicciones)
print("fiabilidad",db_index)
plt.show()
plt.scatter(np.arange(xi.shape[0]), xi[:, 0],c=colores_cluster)