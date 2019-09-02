# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 08:42:54 2019

@author: ramj_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans

dataset = pd.read_csv("C:/Ram Folders/Python/Data/RAM/K-Means-clustering-master/Customers.csv")

X = dataset.iloc[:,3:4]

# Elbow method to find no of clusters

ssq=[]
for i in range(1,11):
    KM=KMeans(n_clusters=i,init='k-means++',random_state=0)
    KM.fit(X)
    ssq.append(KM.inertia_) 
    

plt.plot(range(1,11),ssq)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSQ')
plt.show()

#X=X.value

#Fitting K-MEans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualize the clusters

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='yellow',label='Cluster1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='red',label='Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
