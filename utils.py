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


# Create a small dataset
def create_small_dataset():
    paths = [path_to_test, path_to_train, path_to_train_returns]
    paths_small = [path_to_test_small, path_to_train_small, path_to_train_returns_small]
    
    ntrain = 5000
    ntest = 1000
    numbers = [ntest, ntrain, ntrain]
    
    for path, path_small, number in zip(paths, paths_small, numbers):
        data = pd.read_csv(path)
        data = data.head(number)
        with open(path_small, "w") as f:
            f.write(data.to_csv())
        
    
# Load train data
def load_train(small=False):
    trainPath = path_to_train_small if small else path_to_train
    trainPathReturns = path_to_train_returns_small if small else path_to_train_returns
    
    x_train = pd.read_csv(trainPath)
    y_train = pd.read_csv(trainPathReturns, sep = ',')
        
    y_train_labels = y_train.copy()
    y_train_labels['end_of_day_return'] = 1*(y_train['end_of_day_return']>=0)
        
    return x_train, y_train, y_train_labels


# Load test data
def load_test(small=False):
    testPath = path_to_test_small if small else path_to_test
    x_test = pd.read_csv(testPath)    
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
def submission(prediction, ID = None) :  
    
    if isinstance(prediction,pd.core.frame.DataFrame) :
        prediction.to_csv('predictions.csv', index=False)
    else : 
        pred = pd.DataFrame()
        pred['ID'] = ID
        pred['end_of_day_return'] = prediction
        pred.to_csv('predictions.csv', index=False)
        


# Give some cool bar when we are waiting
def progressBar(value, endvalue, bar_length=50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


if __name__ == "__main__":
    create_small_dataset()
    
