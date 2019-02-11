#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:32:05 2019

@author: evrardgarcelon
"""

from scipy.stats import uniform as sp_uniform

import OneStockOnePredictor as OSOP
import numpy as np
from processing_data import Data
from utils import progressBar

data = Data(small=True, verbose=True)

param_test = {'C': sp_uniform(loc=1, scale=10)}
n_HP_points_to_test = 20
C_tested = []
train_acc = []
test_acc = []

for j in range(n_HP_points_to_test):
    progressBar(j, n_HP_points_to_test - 1)
    C = param_test['C'].rvs()
    C_tested.append(C)
    osop = OSOP(
        data.train.data,
        data.train.labels,
        C=C,
        classifier='LogisticRegression')
    train_acc.append(
        np.mean(
            np.array(
                list(OSOP.score(data.train.data,
                                data.train.labels).values()))))
    test_acc.append(
        np.mean(
            np.array(list(OSOP.score(data.val.data,
                                     data.test.data).values()))))
