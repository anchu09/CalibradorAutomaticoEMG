# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:33:47 2021

@author: Usuario
"""

import matplotlib.pyplot as plt
import numpy as np

xi= np.loadtxt("prueba1_features.txt")

#escalamiento de datos
from sklearn import preprocessing
xi= preprocessing.normalize(xi,axis=0)

#graficamos el dendograma para ver el numero de clusters sugeridos
import scipy.cluster.hierarchy as shc



plt.figure(figsize=[10, 7])

dendrograma= shc.dendrogram(shc.linkage(xi,method='ward'))
#miramos cuantos clusters nos sale (en este caso sale 2)

#definido el numero de cluster se prodece a definir los clusters
from sklearn.cluster import AgglomerativeClustering
algoritmo= AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
#affinity: medidas de distancia, en nuestro caso la euclideana. Es la unica aceptada si utilizamos ward
#linkage: medidas de vinculacion. Guard junta clusters minimizando la varianza 
algoritmo.fit(xi)
etiquetas= algoritmo.labels_


from sklearn.decomposition import PCA
modelo_pca = PCA(n_components = 2, svd_solver='full')
modelo_pca.fit(xi)###### xsinescalar o xescalado
pca = modelo_pca.transform(xi) ###### xsinescalar o xescalado
#Se aplicar la reducción de dimsensionalidad a los centroides
# Se define los colores de cada clúster
colores = ['red','blue']
#Se asignan los colores a cada clústeres
colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]
 
plt.show()  

from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(xi, etiquetas)
print(db_index)
plt.show()
plt.scatter(np.arange(xi.shape[0]), xi[:, 0],c=colores_cluster)