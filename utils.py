#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:15:21 2019

@author: evrardgarcelon
"""

import pandas as pd
from config import *
import numpy as np

# Load train and test data
def load_data(): 
    X_train = pd.read_csv(path_to_train)
    y_train = pd.read_csv(path_to_train_returns,sep = ',')
    X_test = pd.read_csv(path_to_test)

    y_train_labels = y_train.copy()
    y_train_labels['end_of_day_return'] = 1*(y_train['end_of_day_return']>=0)
        
    return X_train, X_test, y_train, y_train_labels


# Split the train dataset into training and validation (keep different date)
def split_dataset(data, split_val=0.1, seed=SEED):
    np.random.seed(seed)

    dates = data["date"].unique().copy()
    n_dates = len(dates)
    all_index = np.arange(n_dates)
    np.random.shuffle(all_index)

    train_index = all_index[int(split_val*n_dates):]
    val_index = all_index[0:int(split_val*n_dates)]

    train = data[data["date"].isin(dates[train_index])]
    val = data[data["date"].isin(dates[val_index])]
    
    return train, val


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


if __name__ == '__main__':
    X_train,X_test,y_train,y_train_labels = load_data()
