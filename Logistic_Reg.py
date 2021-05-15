# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:40:40 2021

@author: admin
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data 
y = dataset.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
from sklearn.preprocessing import MinMaxscaler
mm = MinMaxscaler()
x_train = mm.transform(x_train)
x_test = mm.transform(x_test)
x = mm.transform(x)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

print(log_reg.score(x_train, y_train))
print(log_reg.score(x_test, y_test))
print(log_reg.score(x,y))
y_pred_log = log_reg.predict(x_test)
from sklearn.matrics import classification_report, precision_score, f1_score, confusion_matrics, recall_score
cm_log = confusion_matrics(y_test, y_pred_log)

print(precision_score(y_test, y_pred_log, average = "micro"))

print(recall_score(y_test, y_pred_log, average = "micro"))

print(f1_score(y_test, y_pred_log, average = "micro"))
print(classification_report(y_test, y_pred_log))