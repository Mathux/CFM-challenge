#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:55:14 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from processing_data import Data

from sklearn import model_selection
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgbm


def learning_rate_power_decay(current_iter,gamma = 0.99,base_learning_rate = 0.05):
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


data = Data(small=True, verbose=True)

fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'binary', 
            "eval_set" : [(data.val.data, data.val.labels)],
            'eval_names': ['valid'],
            'callbacks': [lgbm.reset_parameter(learning_rate=learning_rate_power_decay)],
            'verbose': 100,
            'categorical_feature': 'auto'}


param_val ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(20, 100), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

n_HP_points_to_val = 100

clf = lgbm.LGBMClassifier(max_depth=-1, random_state=42, silent=True, metric='None',
                          n_jobs=4, n_estimators=5000, boosting_type = 'gbdt',
                          learning_rate = 0.05, subsampleforbin = 200000, minsplitgain = 0.0)

gs = model_selection.RandomizedSearchCV(estimator=clf,
                                        param_distributions=param_val, 
                                        n_iter=n_HP_points_to_val,
                                        scoring='accuracy',
                                        cv=3,
                                        refit=True,
                                        random_state=42,
                                        verbose=True)

gs.fit(data.train.data, data.train.labels, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

"""
minchildsamples: 20
minchildweight: 0.001
minsplitgain: 0.0
random_state: None
objective: None
class_weight: None
max_depth: -1
num_leaves: 31
reg_alpha: 0.0
subsample_freq: 1
learning_rate: 0.05
n_estimators: 500
colsample_bytree: 0.8
subsample: 0.9
reg_lambda: 0.0
"""
