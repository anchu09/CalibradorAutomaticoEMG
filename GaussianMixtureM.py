# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:05:22 2021

@author: Usuario
"""

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


x1= np.loadtxt("prueba1_features.txt")
x2= np.loadtxt("laura_features.txt")
x3= np.loadtxt("papa_features.txt")
x4= np.loadtxt("alex_features.txt")
x5= np.loadtxt("dani_features.txt")

listaDatos= [x1,x2,x3,x4,x5]
listaresultado=[]
listasobreajustada=[]
contador=0
#sobreajustando datos a proposito 
print("sobreajustados")
for x in listaDatos:
   # print("vuelttaaaaaaaaaaaaaaaaaaa",contador)
   # contador=contador+1
    #escalamiento de datos
    from sklearn import preprocessing
    #normalizamos
    
    #x= preprocessing.normalize(x,axis=0)
    x = preprocessing.minmax_scale(x)
    
    
    #gaussianizamos
    x=np.log(0.0001+x)#ponemos el 0.0001+ para evitar los 0 porque con el log se nos va a inf
    
    #aplicamos reducción de dimensionalidad
    pca= PCA(n_components=1)
    pca.fit(x)
    
    x=pca.transform(x)
    
    
    #creamos el gaussian mixture model
    gmm= GaussianMixture(n_components=2, covariance_type="full", n_init=100)
    gmm.fit(x)
    
    predicciones=gmm.predict(x)
    
  #  for i, pred in enumerate(predicciones):
     #   print("muestra", i, "se encuentra en el cluster:", pred)
        
        
    
    colores = ['red','blue']
    
    colores_cluster = [colores[predicciones[i]] for i in range(len(x))]
     
    plt.show()  
    
    from sklearn.metrics import davies_bouldin_score
    
    db_index = davies_bouldin_score(x, predicciones)
    print("fiabilidad",db_index)
    plt.plot(np.arange(x.shape[0]), x[:, 0])

    plt.scatter(np.arange(x.shape[0]), x[:, 0],c=colores_cluster)
    plt.show()
    listasobreajustada.append(db_index)

#separando datos de prueba y de entrenamiento para evitar el overfitting

print("train test split")
for x in listaDatos:

   # print("vuelttaaaaaaaaaaaaaaaaaaa",contador)
   # contador=contador+1
    #escalamiento de datos
    from sklearn import preprocessing
    #normalizamos
    
    x = preprocessing.minmax_scale(x)
   
    #gaussianizamos
    x=np.log(0.0000001+x)#ponemos el 0.0001+ para evitar los 0 porque con el log se nos va a inf
    
    #aplicamos reducción de dimensionalidad
    pca= PCA(n_components=1)
 
    from sklearn.model_selection import train_test_split
    X_train, xtest = train_test_split(x,test_size=0.3)#importante el orden para que no de errores

    
    #creamos el gaussian mixture model
    pca.fit(X_train)
    gmm= GaussianMixture(n_components=2, covariance_type="full", n_init=100)
    X_train = pca.fit_transform(X_train)
    gmm.fit(X_train)
    
    xtest = pca.transform(xtest)
    predicciones=gmm.predict(xtest)
    
   # for i, pred in enumerate(predicciones):
       # print("muestra", i, "se encuentra en el cluster:", pred)
        
        
    
    colores = ['red','blue']
    
    colores_cluster = [colores[predicciones[i]] for i in range(len(xtest))]
     
    plt.show()  
    
    from sklearn.metrics import davies_bouldin_score
    
    db_index = davies_bouldin_score(xtest, predicciones)
    import seaborn as sns
    print("fiabilidad",db_index)
    #sns.kdeplot(xtest[np.array(colores_cluster) == "red", ].flatten(), bw=0.5)
    #sns.kdeplot(xtest[np.array(colores_cluster) == "blue", ].flatten(), bw=0.5)
    plt.hist(xtest[np.array(colores_cluster) == "red", ].flatten())
    plt.hist(xtest[np.array(colores_cluster) == "blue", ].flatten())
    plt.show()
    
    plt.figure()
    plt.plot(np.arange(xtest.shape[0]), xtest[:, 0])

    plt.scatter(np.arange(xtest.shape[0]), xtest[:, 0],c=colores_cluster)
    plt.show()

    listaresultado.append(db_index)
    
    
    
    
""" 
   tabla_Rendimiento = pd.DataFrame()
tabla_Rendimiento['Set','No preproces','Gauss','Norm','Red_dim','Gauss+Norm','Gauss+red_dim','Norm+Red_dim','Gauss+Norm+Red_dim'] = None
nueva_Fila={'Set':'x1', 
            'No preproces': puntuacion(x1, False, False, False),
            'Gauss':puntuacion(x1, True, False, False),
            'Norm':puntuacion(x1, False, True, False),
            'Red_dim':puntuacion(x1, False, False, True),
            'Gauss+Norm':puntuacion(x1, True, True, False),
            'Gauss+red_dim':puntuacion(x1, True, False, True),
            'Norm+Red_dim':puntuacion(x1, False, True, True),
            'Gauss+Norm+Red_dim':puntuacion(x1, True, True, True)}
tabla_Rendimiento=tabla_Rendimiento.append(nueva_Fila,ignore_index=True)
print(tabla_Rendimiento)

"""
