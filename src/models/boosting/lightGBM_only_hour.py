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

data.train.data = data.train.data.drop(['ID','date'],axis = 1)
data.val.data = data.val.data.drop(['ID','date'],axis = 1)

not_return_cols = [col for col in data.train.data.columns if col.endswith(':00')]
data.train.data = data.train.data[not_return_cols]

not_return_cols = [col for col in data.val.data.columns if col.endswith(':00')]
data.val.data = data.val.data[not_return_cols]


test_id = data.test.data['ID']
data.test.data = data.test.data.drop(['ID','date'],axis = 1)
not_return_cols = [col for col in data.test.data.columns if col.endswith(':00')]
data.test.data = data.test.data[not_return_cols]

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

param_test = {
    'num_leaves': sp_randint(6, 50),
    'min_child_samples': sp_randint(50, 130),
    'min_child_weight': [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
    'subsample': sp_uniform(loc=0.9, scale=0.1),
    'colsample_bytree': sp_uniform(loc=0.7, scale=0.3),
    'reg_alpha': [2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7],
    'reg_lambda': [1, 2, 3, 0, 0.5, 0.3, 0.1, 0.01, 0.001]
}

n_HP_points_to_test = 500

clf = lgbm.LGBMClassifier(
    max_depth=-1,
    random_state=42,
    silent=True,
    metric='None',
    n_jobs=4,
    n_estimators=5000,
    boosting_type='gbdt',
    learning_rate=0.05)

gs = model_selection.RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_test,
    n_iter=n_HP_points_to_test,
    scoring='accuracy',
    cv=3,
    refit=True,
    random_state=42,
    verbose=True,
)

gs.fit(data.train.data, train_labels, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_,
                                                       gs.best_params_))

clf = lgbm.LGBMClassifier(importance_type = 'gain',**gs.best_params_)
clf.fit(data.train.data, train_labels, **fit_params)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)


if False:
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,data.train.data.columns)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    #plt.show()
    plt.savefig('lgbm_importances-01.png')


#preds = gs.predict(data.test.data)
#from src.tools.utils import submission
#submission(preds, test_id)

# import shap
# sht = shap.TreeExplainer(clf)
# shval = sht.shap_values(data.train.data)
# shap.summary_plot(shval, data.train.data)
