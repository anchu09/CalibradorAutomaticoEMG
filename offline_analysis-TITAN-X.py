import numpy as np
from sklearn.cluster import KMeans


xi = np.loadtxt("prueba1_features.txt")

#me quedo solo con la columna importante
import matplotlib.pyplot as plt
#plt.scatter(range(0,len(xi)),xi)


def algoritmoCodo():
    inercia=[]
    for i in range(1,20):
        algoritmo= KMeans(n_clusters=i, init ='k-means++', max_iter=300, n_init=10)
        algoritmo.fit(xi)
        #para cada k se calcula la suma total del cuadrado dentro del cluster
        inercia.append(algoritmo.inertia_)
    #trazamos la curva de la suma de errores cuadráticos
    plt.figure(figsize=[10,6])
    plt.title('método del codo')
    plt.xlabel('nº de cluster')
    plt.ylabel('inercia')
    plt.plot(list(range(1,20)), inercia, marker='o')
    plt.show()
    
#por curiosidad miramos cuantos clusters nos sugiere el algoritmo del codo
#algoritmoCodo()


#pasamos todo a un array bidimensional para que podamos hacer el fit
#index=np.array(range(0,len(xi)))
#c=np.empty((xi.size+index.size,),dtype=xi.dtype)
#datatonormalize = np.reshape(c, (-1,2))

#xi=xi.reshape(len(xi),1)
#escalamiento de los datos
from sklearn import preprocessing

# defalult (axis =1) en nuestro caso axis=0 para el eje y
#data_escalada= preprocessing.Normalizer().fit_transform(datatonormalize)
data_escalada= preprocessing.normalize(xi,axis=0)


#determinamos las variables a evaluar 
xescalado=data_escalada.copy()
xsinescalar=xi.copy()


algoritmo= KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10)
algoritmo.fit(xescalado)##### xsinescalar o xescalado
centroides, etiquetas= algoritmo.cluster_centers_, algoritmo.labels_
muestra_predicciones=algoritmo.predict(data_escalada)####datatornomalize o data_escalada

for i, pred in enumerate(muestra_predicciones):
    print("muestra", i, "se encuentra en el cluster:", pred)

  
from sklearn.decomposition import PCA
modelo_pca = PCA(n_components = 2, svd_solver='full')
modelo_pca.fit(xescalado)###### xsinescalar o xescalado
pca = modelo_pca.transform(xescalado) ###### xsinescalar o xescalado
#Se aplicar la reducción de dimsensionalidad a los centroides
centroides_pca = modelo_pca.transform(centroides)
# Se define los colores de cada clúster
colores = ['red','blue']
#Se asignan los colores a cada clústeres
colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]
#Se grafica los componentes PCA
plt.scatter(pca[:, 0], pca[:, 1], c = colores_cluster, 
            marker = 'o',alpha = 0.4)
#Se grafican los centroides
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
            marker = 'x', s = 100, linewidths = 3, c = colores)
#Se guadan los datos en una variable para que sea fácil escribir el código
xvector = modelo_pca.components_[0] * max(pca[:,0])
yvector = modelo_pca.components_[1] * max(pca[:,1])


 
plt.show()  


plt.figure()
plt.scatter(np.arange(xi.shape[0]), xi[:, 0],c=colores_cluster)

#evaluacion del algoritmo
from sklearn.metrics import davies_bouldin_score
db_index = davies_bouldin_score(xi,etiquetas)
print(db_index)