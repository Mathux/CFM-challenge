#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:55:14 2019

@author: evrardgarcelon, mathispetrovich
"""

import lightgbm as lgbm
import numpy as np
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn import model_selection

from src.tools.dataloader import Data

data = Data(small=False, verbose=True, aggregate=False, ewma=False)

data.train.data = data.train.data.drop(['ID', 'date'], axis=1)
data.val.data = data.val.data.drop(['ID', 'date'], axis=1)
test_id = data.test.data['ID']
data.test.data = data.test.data.drop(['ID', 'date'], axis=1)
train_labels = data.train.labels['end_of_day_return']
val_labels = data.val.labels['end_of_day_return']


def learning_rate_power_decay(current_iter,
                              gamma=0.99,
                              base_learning_rate=0.05):
    lr = base_learning_rate * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


fit_params = {
    "early_stopping_rounds": 30,
    "eval_metric": 'binary',
    "eval_set": [(data.val.data, val_labels)],
    'eval_names': ['valid'],
    'callbacks':
    [lgbm.reset_parameter(learning_rate=learning_rate_power_decay)],
    'verbose': 100,
    'categorical_feature': 'auto'
}

params = {
    'colsample_bytree': 0.7295359988236723,
    'min_child_samples': 67,
    'min_child_weight': 10000.0,
    'num_leaves': 45,
    'reg_alpha': 5,
    'reg_lambda': 0,
    'subsample': 0.9638922245305552
}

clf = lgbm.LGBMClassifier(importance_type='gain', **params)
clf.fit(data.train.data, train_labels, **fit_params)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

if False:
    feature_imp = pd.DataFrame(
        sorted(zip(clf.feature_importances_, data.train.data.columns)),
        columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    #plt.show()
    plt.savefig('lgbm_importances-01.png')

preds = clf.predict(data.test.data)
from src.tools.utils import submission
submission(preds, test_id)

# import shap
# sht = shap.TreeExplainer(clf)
# shval = sht.shap_values(data.train.data)
# shap.summary_plot(shval, data.train.data, show=True)


# fig = plt.gcf()
# plt.savefig("shaptest.png")
