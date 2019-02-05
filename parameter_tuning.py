#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:32:05 2019

@author: evrardgarcelon
"""

from utils import *
import numpy as np
from OneStockOnePredictor import *

from sklearn import model_selection
from scipy.stats import uniform as sp_uniform

X_train,X_test,y_train,y_train_labels = load_data()
add_features(X_train)
add_features(X_test)

cols_to_use = [c for c in X_train.columns if not c.endswith(':00')]
X_train,X_test = X_train[cols_to_use],X_test[cols_to_use]

train, test, train_labels, test_labels = model_selection.train_test_split(X_train, y_train_labels, test_size=0.10, random_state=42)

param_test ={'C': sp_uniform(loc = 1,scale = 10)}
n_HP_points_to_test = 20
C_tested = []
train_acc = []
test_acc = []


for j in range(n_HP_points_to_test) :
    progressBar(j,n_HP_points_to_test-1)
    C = param_test['C'].rvs()
    C_tested.append(C)
    OSOP = OneStockOnePredictor(train,train_labels,C = C,classifier = 'LogisticRegression')
    train_acc.append(np.mean(np.array(list(OSOP.score(train,train_labels).values()))))
    test_acc.append(np.mean(np.array(list(OSOP.score(test,test_labels).values()))))
