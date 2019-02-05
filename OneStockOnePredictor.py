#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:31:54 2019

@author: evrardgarcelon
"""

import numpy as np
import pandas as pd
from utils import submission, progressBar
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class OneStockOnePredictor(object) :
    def __init__(self,train,train_labels,C = 1,progress_bar = False, classifier = 'LogisticRegression') :
        cols = [col for col in train.columns if not col.endswith(':00')]
        train = train[cols]
        eqt_code = train['eqt_code'].unique()
        stocks_done = 0
        nb_stocks = len(eqt_code)
        self.predictors = {}
        train = train.merge(train_labels,on = 'ID',how = 'inner')
        self.C = C
        self.progress_bar = progress_bar
        
        nb_null_classes = 0
        
        for code in eqt_code :     
            if progress_bar :
                progressBar(stocks_done,nb_stocks)
            data_code = train[train['eqt_code']==code].sort_values(by = 'ID')
            labels = data_code['end_of_day_return']
            
            data_code = data_code.drop(['eqt_code','end_of_day_return','ID'],axis =1)
            
            if np.sum(labels == 1) > 0 and np.sum(labels == 0) > 0 :
                
                if classifier == 'LogisticRegression' :
                    
                    self.predictors[code] = LogisticRegression(solver = 'lbfgs', max_iter = 500, C = self.C, penalty = 'l2').fit(data_code,labels)
                    
                elif classifier == 'SVM' :
                    
                    self.predictors[code] = SVC(C = self.C,gamma = 'scale').fit(data_code,labels)
                    
                elif classifier == 'KNN' :
                    
                    self.predictors[code] = KNeighborsClassifier(n_neighbors = self.C).fit(data_code,labels)
                    
                else : 
                    print('Error Classifier')
                    break
                    
            else :
                nb_null_classes +=1
                if np.sum(labels == 1) > 0 :
                    self.predictors[code] = DummyClassifier(strategy= 'constant',constant=1).fit(data_code,labels)
                else : 
                    self.predictors[code] = DummyClassifier(strategy= 'constant',constant=0).fit(data_code,labels)

            stocks_done +=1
        
    
    def predict(self,X) :
        
        cols = [col for col in X.columns if not col.endswith(':00')]
        X = X[cols]
        predicted_labels = pd.DataFrame(columns = ['ID','end_of_day_return'])
        eqt_code = X['eqt_code'].unique()
        stocks_done = 0
        nb_stocks = len(eqt_code)
        
        for code in eqt_code :
            if self.progress_bar :
                progressBar(stocks_done,nb_stocks)
            data_code = X[X['eqt_code']==code].sort_values(by = 'ID')
            temp = pd.DataFrame()
            temp['ID'] = data_code['ID']
            data_code = data_code.drop(['ID','eqt_code'],axis =1)
            label = self.predictors[code].predict(data_code)
            temp['end_of_day_return'] = label
            predicted_labels = predicted_labels.append(temp)
            stocks_done +=1
        
        predicted_labels.sort_values(by = 'ID')
        
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
            
            if self.progress_bar :
                progressBar(stocks_done,nb_stocks)
            data_code = X[X['eqt_code']==code].sort_values(by = 'ID')
            labels = data_code['end_of_day_return']
            data_code = data_code.drop(['end_of_day_return','ID','eqt_code'],axis =1)
            self.accuracy[code] = self.predictors[code].score(data_code,labels)
            stocks_done +=1
        
        return self.accuracy
