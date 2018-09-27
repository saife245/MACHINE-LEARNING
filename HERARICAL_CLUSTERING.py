# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:39:32 2018

@author: MD SAIF UDDIN
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data set
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

#using the dendogram to find the optimal number of cluster
import scipy.cluster.hierarchy as sch  #scipy is tool to do hierarical cluster and program
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))#linkage is a algorithm itself for hierarical clustering
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

#fittting the heirarical clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the cluster 
plt.scatter(X[y_hc ==0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'carefull')
plt.scatter(X[y_hc ==1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc ==2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'target')
plt.scatter(X[y_hc ==3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'careless')
plt.scatter(X[y_hc ==4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'sensible')
plt.title('Cluster of client')
plt.xlabel('annual income ')
plt.ylabel('spending score')
plt.legend()
plt.show()