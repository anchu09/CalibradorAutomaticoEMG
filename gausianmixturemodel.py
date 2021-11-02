# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:05:22 2021

@author: Usuario
"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score


x1= np.loadtxt("prueba1_features.txt")
x2= np.loadtxt("laura_features.txt")
x3= np.loadtxt("papa_features.txt")
x4= np.loadtxt("alex_features.txt")
x5= np.loadtxt("dani_features.txt")

lista_Datos= [x1,x2,x3,x4,x5]

def reescalamiento(datos):
    return preprocessing.minmax_scale(datos)

def gaussianizar(datos):
    return np.log(0.0000001+datos)

def puntuacion(x,reescalar, gaus, red_dim,t_size):
    if (reescalar==True):
        x = reescalamiento(x)
        
    if (gaus==True):
        x = gaussianizar(x) 
  
    if (red_dim==True):
        
        pca= PCA(n_components=1)
    
        X_train, X_test = train_test_split(x,test_size=t_size)
           
        gmm= GaussianMixture(n_components=2, covariance_type="full", n_init=100)
        
        X_train = pca.fit_transform(X_train)
        
        gmm.fit(X_train)
        
        X_test = pca.transform(X_test)
    
    else:
    
        X_train, X_test = train_test_split(x,test_size=t_size)
           
        gmm= GaussianMixture(n_components=2, covariance_type="full", n_init=100)
                
        gmm.fit(X_train)
        
        
    predicciones=gmm.predict(X_test)
 
    colores = ['red','blue']
    
    colores_cluster = [colores[predicciones[i]] for i in range(len(X_test))]
     
    plt.show()  
        
    db_index = davies_bouldin_score(X_test, predicciones)
   
    plt.scatter(np.arange(X_test.shape[0]), X_test[:, 0],c=colores_cluster)
    
    plt.show()

    return db_index


datos_Comparacion = {'No_Prep': puntuacion(x1, False, False, False,0.3),
            'Ga':puntuacion(x1, True, False, False,0.3),
            'Nor':puntuacion(x1, False, True, False,0.3),
            'R_d':puntuacion(x1, False, False, True,0.3),
            'G+N':puntuacion(x1, True, True, False,0.3),
            'G+r_d':puntuacion(x1, True, False, True,0.3),
            'N+r_d':puntuacion(x1, False, True, True,0.3),
            'G+N+R_d':puntuacion(x1, True, True, True,0.3)}

plt.bar(datos_Comparacion.keys(), datos_Comparacion.values())
#observaciones: lo que mejor funciona es normalizar y reducir la dimensionalidad = N+r_d
#pero esta opción no funciona en algunos sets de datos como el 2 -> puntuacion(x2, False, True, True) PETA
#aunque G+N+R_d es peor pero como la diferencia es pequeña, cogemos la opcion puntuacion(x2, True, True, True) 


#para todas las muestras:
lista_Resultados=[]
for x in lista_Datos:
 lista_Resultados.append(puntuacion(x, True, True, True,0.3)) 






#test: quiero ver con qué tamaño funciona mejor

#lista_x1=[]
#lista_x2=[]
#lista_x3=[]
#for i in range(9):
   # lista_x1.append(puntuacion(x1, True, True, True,i/10+0.1)) 
   # lista_x2.append(puntuacion(x1, True, True, True,i/10+0.1)) 
    #lista_x3.append(puntuacion(x1, True, True, True,i/10+0.1)) 
    
#plt.plot(lista_x1)
#plt.plot(lista_x2)
#plt.plot(lista_x3)

#concluision: con ninguno en especial porque el score depende del conjunto
#de datos que es aleatorio. así que lo dejo en 0.3 
  
