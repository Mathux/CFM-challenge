#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:31:54 2019

@author: evrardgarcelon
"""

import numpy as np
import pandas as pd
from data_exploration import submission,progressBar
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


    

class OneStockOnePredictor(object) :
    
    def __init__(self,train,train_labels) :
        
        
        cols = [col for col in train.columns if not col.endswith(':00')]
        train = train[cols]
        eqt_code = train['eqt_code'].unique()
        stocks_done = 0
        nb_stocks = len(eqt_code)
        self.predictors = {}
        train = train.merge(train_labels,on = 'ID',how = 'inner')
        
        for code in eqt_code :        
            progressBar(stocks_done,nb_stocks)
            self.predictors[code] = LogisticRegression(solver = 'lbfgs',max_iter = 1000)
            data_code = train[train['eqt_code']==code].sort_values(by = 'ID')
            labels = data_code['end_of_day_return']
            
            data_code = data_code.drop(['end_of_day_return','ID'],axis =1)
            
            if np.sum(labels == 1) > 0 and np.sum(labels == 0) > 0 :
                self.predictors[code].fit(data_code,labels)
            else :
                if np.sum(labels == 1) > 0 :
                    self.predictors[code] = DummyClassifier(strategy= 'constant',constant=1)
                else : 
                    self.predictors[code] = DummyClassifier(strategy= 'constant',constant=0)
                
                self.predictors[code].fit(data_code,labels)

            stocks_done +=1
            
            
    def predict(self,X) :
        
        cols = [col for col in X.columns if not col.endswith(':00')]
        X = X[cols]
        predicted_labels = pd.DataFrame()
        predicted_labels['ID'] = X['ID']
        predicted_labels['end_of_day_return'] = np.zeros(len(X['ID']),dtype = 'int')
        eqt_code = X['eqt_code'].unique()
        stocks_done = 0
        nb_stocks = len(eqt_code)
        
        for code in eqt_code :
            progressBar(stocks_done,nb_stocks)
            data_code = X[X['eqt_code']==code].sort_values(by = 'ID')
            ids = data_code['ID'].unique()
            data_code = data_code.drop(['ID'],axis =1)
            label = self.predictors[code].predict(data_code)
            for j in range(len(ids)) : 
                predicted_labels.loc[predicted_labels['ID'] == ids[j], 'end_of_day_return'] = label[j]
            stocks_done +=1
        return predicted_labels
    
    
    def score(self,X,y) :
        
        cols = [col for col in X.columns if not col.endswith(':00')]
        X = X[cols]
        eqt_code = X['eqt_code'].unique()
        stocks_done = 0
        nb_stocks = len(eqt_code)
        self.accuracy = {}
        X = X.merge(y,on = 'ID',how = 'inner')

        
        for code in eqt_code :
            
            progressBar(stocks_done,nb_stocks)
            data_code = X[X['eqt_code']==code].sort_values(by = 'ID')
            labels = data_code['end_of_day_return']
            data_code = data_code.drop(['end_of_day_return','ID'],axis =1)
            self.accuracy[code] = self.predictors[code].score(data_code,labels)
            stocks_done +=1
        
        return self.accuracy
        
        
 #%%       
    

        