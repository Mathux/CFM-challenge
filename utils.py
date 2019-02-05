#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:15:21 2019

@author: evrardgarcelon, mathispetrovich
"""

import pandas as pd
from config import *
import numpy as np
import sys

# Load train data
def load_train():
    x_train = pd.read_csv(path_to_train)
    y_train = pd.read_csv(path_to_train_returns,sep = ',')

    y_train_labels = y_train.copy()
    y_train_labels['end_of_day_return'] = 1*(y_train['end_of_day_return']>=0)
        
    return x_train, y_train, y_train_labels


# Load test data
def load_test():
    x_test = pd.read_csv(path_to_test)
    return x_test


# Split the train dataset into training and validation (keep different date)
def split_dataset(data, labels, split_val=0.1, seed=SEED):
    np.random.seed(seed)
    
    data = data.merge(labels, on = 'ID')

    dates = data["date"].unique().copy()
    n_dates = len(dates)
    all_index = np.arange(n_dates)
    np.random.shuffle(all_index)

    train_index = all_index[int(split_val*n_dates):]
    val_index = all_index[0:int(split_val*n_dates)]

    train = data[data["date"].isin(dates[train_index])]
    val = data[data["date"].isin(dates[val_index])]
    
    train_labels = train[['ID','end_of_day_return']]
    val_labels = val[['ID','end_of_day_return']]
    
    train = train.drop('end_of_day_return',axis = 1)
    val = val.drop('end_of_day_return',axis = 1)
    
    return train, val, train_labels, val_labels


# Tools to give the csv format
def submission(prediction) :    
    prediction.to_csv('predictions.csv', index=False)


# Give some cool bar when we are waiting
def progressBar(value, endvalue, bar_length=50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
