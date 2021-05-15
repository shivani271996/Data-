# -*- coding: utf-8 -*-
"""
Created on Fri May 14 21:45:28 2021

@author: admin
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data 
y = dataset.target
from sklearn.cluster import KMeans
wcv = []
for i in range(1, 16):
    km = KMeans(n_clusters = i)
    km.fit(x)
    
    wcv.append(km.insertia_)
plt.plot(range(1, 16), wcv)
plt.show()
km = KMeans(n_clusters = 3)
y_pred = km.fit_predict(x)
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1])
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1])
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1])
plt.title("original")
plt.show()

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1])
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1])
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1])
plt.title("predicted")
plt1 = plt.figure()
plt.show()

plt.scatter(x[y_pred == 0, 2], x[y_pred == 0, 3])
plt.scatter(x[y_pred == 1, 2], x[y_pred == 1, 3])
plt.scatter(x[y_pred == 2, 2], x[y_pred == 2, 3])
plt.title("original")
plt.show()

  
