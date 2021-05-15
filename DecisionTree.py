# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:11:46 2021

@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
dataset = load_wine()
x = dataset.data 
y = dataset.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
from sklearn.preprocessing import MinMaxscaler
mm = MinMaxscaler()
x_train = mm.transform(x_test)
x = mm.transform(x)
from sklearn.neighbors import DecisionTreeClassifier
dtc= DecisionTreeClassifier(max_depth = 10)
dtc.fit(x_train, y_train)
print(dtc.score(x_train, y_train))
print(dtc.score(x_test, y_test))
print(dtc.score(x,y))
y_pred_dtc = dtc.predict(x_test)

from sklearn.matrics import classification_report, precision_score, f1_score, confusion_matrix, recall_score
cm_dtc = confusion_matrix(y_test, y_pred_dtc)

print(cm_dtc)
print(precision_score(y_test, y_pred_dtc, average = "micro"))

print(recall_score(y_test, y_pred_dtc, average = "micro"))

print(f1_score(y_test, y_pred_dtc, average = "micro"))
print(classification_report(y_test, y_pred_dtc, average = "micro"))
from sklearn import tree
text_representation = tree.export_text(dtc)
print(text_representation)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dtc,
               feature_name = dataset.feature_names,
               class_names=dataset.target_names,
               filled = True)
plt.savefig("imagename.png")