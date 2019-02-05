#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:42:18 2019

@author: evrardgarcelon
"""

import numpy as np
import pandas as pd
from utils import submission, progressBar
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class Stacking(object) :
    def __init__(self,train, train_labels, weak_classifiers = [], meta_classifier = "LogisticRegression",progress_bar = False) :
        
        cols = [col for col in train.columns if not col.endswith(':00')]
        train = train[cols]
        train = train.drop('ID',axis = 1)
        self.nb_clfs = len(weak_classifiers)
        self.progress_bar = progress_bar
        self.weak_classifiers = weak_classifiers
        new_dataset = pd.DataFrame()
        
        print('Fitting weak classifiers')
        for j in range(len(self.weak_classifiers)) : 
            
            if progress_bar :
                progressBar(j,self.nb_clfs)
                
            self.weak_classifiers[j].fit(train,train_labels['end_of_day_return'])
            new_dataset[j] = self.weak_classifiers[j].predict_proba(train)[:,0]

                
        print('Fitting the meta-classifier')
        if meta_classifier == 'LogisticRegression' :
            
            self.meta_clf = LogisticRegression(C = 1,solver = 'lbfgs',max_iter = 1000)
            self.meta_clf.fit(new_dataset,train_labels['end_of_day_return'])
            
        elif meta_classifier == 'SVM' :
            
            self.meta_clf = SVC(C = 1)
            self.meta_clf.fit(new_dataset,train_labels['end_of_day_return'])
            
        elif meta_classifier == 'KNN' :
            
            self.meta_clf = SVC(C = 1)
            self.meta_clf.fit(new_dataset,train_labels['end_of_day_return'])
            
        else : 
            print('Error Classifier')
                
          
    def predict(self,X) :
        
        cols = [col for col in X.columns if not col.endswith(':00')]
        X = X[cols]
        predicted_labels = pd.DataFrame(columns = ['ID','end_of_day_return'])
        predicted_labels['ID'] = X['ID']
        new_X = pd.dataframe()
        
        for j in range(len(self.weak_classifier)) :
            if self.progress_bar :
                progressBar(j,self.nb_clfs)
            new_X[j] = self.weak_classifiers[j].predict_proba(X)[:,0]
        predicted_labels['end_of_day_return'] = self.meta_clf.predict(new_X)    
        
        return predicted_labels
    
    
    def score(self,X,y) :
        
        cols = [col for col in X.columns if not col.endswith(':00')]
        X = X[cols]
        predicted_labels = pd.DataFrame(columns = ['ID','end_of_day_return'])
        predicted_labels['ID'] = X['ID']
        new_X = pd.dataframe()
        
        for j in range(len(self.weak_classifier)) :
            if self.progress_bar :
                progressBar(j,self.nb_clfs)
            new_X[j] = self.weak_classifiers[j].predict_proba(X)[:,0]  
        
        return self.meta_clf.score(new_X,y['end_of_day_return'])

class Averaging(object) :
    
    def __init__(self,train,train_labels,base_classifiers = [],weights = [], progress_bar = False):
        
        if len(weights) == 0:
            self.weights = np.ones(len(base_classifiers))/len(base_classifiers)
        else :
            self.weights = weights
        
        self.base_classifiers = base_classifiers
        self.progress_bar = progress_bar
        print('Fitting base classifiers')
        for j in range(len(self.base_classifiers)) :
            if self.progress_bar :
                progressBar(j,len(self.weights))
            self.base_classifiers[j].fit(train,train_labels['end_of_day_return'])
        
    def predict(self,X) :
        predicted_proba = []
        for j in range(len(self.base_classifiers)) :
            predicted_proba.append(self.base_classifiers[j].predict_proba(X)[:,0])
        predicted_proba = np.array(predicted_proba)
        final_proba = 0
        for j in range(len(self.base_classifiers)) :
            final_proba = final_proba + self.weights[j]*predicted_proba[j]
        predicted_labels = final_proba[final_proba < 0.5]
        
        df = pd.DataFrame()
        df['ID'] = X['ID']
        df['end_of_day_return'] = predicted_labels
        
        return predicted_labels
    
    def score(self,X,y) :
                    
        predicted_proba = []
        for j in range(len(self.base_classifiers)) :
            predicted_proba.append(self.base_classifiers[j].predict_proba(X)[:,0])
        predicted_proba = np.array(predicted_proba)
        final_proba = 0
        for j in range(len(self.base_classifiers)) :
            final_proba = final_proba + self.weights[j]*predicted_proba[j]
        predicted_labels = final_proba[final_proba < 0.5]
        
        accuracy = np.mean(predicted_labels == y['end_of_day_return'])
        
        return accuracy
            
        
        
