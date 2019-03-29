#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:49:39 2019

@author: evrardgarcelon
"""

import pandas as pd

class VerySimplePredictor(object):
    def __init__(self):
        pass
    
    def predict(self, X):
        predicted_labels = pd.DataFrame(columns=['ID', 'end_of_day_return'])
        predicted_labels['ID'] = X['ID']
        col = 'avg_return_date_eqt'
        X = X[col]
        predictions = 1*(X<0)
        predicted_labels['end_of_day_return'] = predictions
        return predicted_labels


if __name__ == '__main__' :
    from src.tools.dataloader import Data
    from src.tools.utils import submission
    import numpy as np
    data = Data(verbose = True)
    clf = VerySimplePredictor()
    train_prediction = clf.predict(data.train.data)
    print("Accuracy : ", np.sum(data.train.labels['end_of_day_return'].values == train_prediction['end_of_day_return'].values)/len(data.train.labels))
    test_prediction = clf.predict(data.test.data)
    submission(test_prediction)