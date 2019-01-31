
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:15:21 2019

@author: evrardgarcelon
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt

path_to_test = "/Users/evrardgarcelon/Desktop/CFM_challenge/input_test.csv"
path_to_train = "/Users/evrardgarcelon/Desktop/CFM_challenge/input_training.csv"
path_to_train_returns = "/Users/evrardgarcelon/Desktop/CFM_challenge/output_training.csv"

def load_data() : 
        X_train = pd.read_csv(path_to_train)
        y_train = pd.read_csv(path_to_train_returns,sep = ',')
        X_test = pd.read_csv(path_to_test)
        
        #Cleaning

        #X_train['09:30:00'].fillna(0,inplace = True)
        #X_train.iloc[:,3:] = X_train.iloc[:,3:].interpolate(axis=1)
        
        #X_test['09:30:00'].fillna(0,inplace = True)
        #X_test.iloc[:,3:] = X_test.iloc[:,3:].interpolate(axis=1)
        
        
        y_train_labels = y_train.copy()
        y_train_labels['end_of_day_return'] = 1*(y_train['end_of_day_return']>=0)

        #Feature Engineering

        #X_train['mean return'] = X_train.iloc[:,3:].mean(axis = 1)
#        X_train['median return'] = X_train.iloc[:,3:].median(axis = 1)
#        X_train['max return'] = X_train.iloc[:,3:].max(axis = 1)
#        X_train['min return'] = X_train.iloc[:,3:].min(axis = 1)
#        X_train['drawdown'] = X_train['max return'] - X_train['min return']
#        
#        X_test['mean return'] = X_test.iloc[:,3:].mean(axis = 1)
#        X_test['median return'] = X_test.iloc[:,3:].median(axis = 1)
#        X_test['max return'] = X_test.iloc[:,3:].max(axis = 1)
#        X_test['min return'] = X_test.iloc[:,3:].min(axis = 1)
#        X_test['drawdown'] = X_test['max return'] - X_test['min return'] 
#        
#        market_return = X_train.groupby('date').mean().iloc[:,3:].mean(axis = 1).reset_index()
#        X_train = X_train.merge(market_return,how = 'inner',on = 'date')
#        X_train = X_train.rename({0 : 'market return'},axis = 'columns')
#        
#        market_return = X_test.groupby('date').mean().iloc[:,3:].mean(axis = 1).reset_index()
#        X_test = X_test.merge(market_return,how = 'inner',on = 'date')
#        X_test = X_test.rename({0 : 'market return'},axis = 'columns')
        
#        
#        X_train = X_train.sort_values(by = 'ID').reset_index().drop('index',axis = 1)
#        X_test = X_test.sort_values(by = 'ID').reset_index().drop('index',axis = 1)
        
        return X_train,X_test,y_train,y_train_labels

def submission(prediction) :
    
    prediction.to_csv('predictions.csv', index=False)
    pass

def progressBar(value, endvalue, bar_length=50):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n Progress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    pass
    
def feature_engineering(train,test) :
    
    return_cols = [col for col in train.columns if col.endswith(':00')]
    train['return_nan'] = train.isna().sum(axis = 1)
    test['return_nan'] = test.isna().sum(axis = 1)
        
    train['avg_return_date_eqt'] = train[return_cols].mean(axis = 1)
    test['avg_return_date_eqt'] = test[return_cols].mean(axis = 1)

    train['var_return_date_eqt'] = train[return_cols].var(axis = 1)
    test['var_return_date_eqt'] = test[return_cols].var(axis = 1)

    train['skew_return_date_eqt'] = train[return_cols].skew(axis = 1)
    test['skew_return_date_eqt'] = test[return_cols].skew(axis = 1)

    train['kurt_return_date_eqt'] = train[return_cols].kurt(axis = 1)
    test['kurt_return_date_eqt'] = test[return_cols].kurt(axis = 1)  
    
    train,test = group_by_date_countd(train,return_cols),group_by_date_countd(test,return_cols) 
    train,test = group_by_product_countd(train,return_cols),group_by_product_countd(test,return_cols)
    
    train['tot_return_eqt_date'] = train[return_cols].sum(axis = 1)
    test['tot_return_eqt_date'] = test[return_cols].sum(axis = 1)
    
    stock_correlation_train = market_correlation(train)
    stock_correlation_train = stock_correlation_train.replace(to_replace = 1.0, value = 0)
    stock_correlation_test = market_correlation(test)
    stock_correlation_test = stock_correlation_test.replace(to_replace = 1.0, value = 0)
    
    temp_train = (train.groupby('eqt_code')['tot_return_eqt_date'].mean()*stock_correlation_train/stock_correlation_train.sum(axis = 1)).sum()
    temp_test = (test.groupby('eqt_code')['tot_return_eqt_date'].mean()*stock_correlation_test/stock_correlation_test.sum(axis = 1)).sum()
    
    train.set_index(['eqt_code'],inplace = True)
    test.set_index(['eqt_code'],inplace = True)
    
    train['market_feature'] = temp_train
    test['market_feature'] = temp_test
    
    train.reset_index(inplace = True)
    test.reset_index(inplace = True)
    
    return train,test


def group_by_date_countd(all_data,return_cols):
    groupby_col = "date"
    unique_products = all_data.groupby([groupby_col])["eqt_code"].nunique()
    avg_market_return = all_data.groupby([groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby([groupby_col])['var_return_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_product"] = unique_products.astype('uint16')
    all_data["avg_market_return_date"] = avg_market_return.astype('float64')
    all_data["var_marlet_return_date"] = var_market_return.astype('float64')
    all_data.reset_index(inplace=True)
    return all_data


def group_by_product_countd(all_data,return_cols):
    groupby_col = "eqt_code"
    unique_date = all_data.groupby([groupby_col])["date"].nunique()
    avg_market_return = all_data.groupby([groupby_col])['avg_return_date_eqt'].mean()
    var_market_return = all_data.groupby([groupby_col])['var_return_date_eqt'].mean()
    all_data.set_index([groupby_col], inplace=True)
    all_data["countd_date"] = unique_date.astype('uint16')
    all_data["avg_market_return_eqt"] = avg_market_return.astype('float64')
    all_data["var_market_return_eqt"] = var_market_return.astype('float64')
    all_data.reset_index(inplace=True)
    return all_data

def market_correlation(data) :
    
    df = pd.pivot_table(data[['date','eqt_code','tot_return_eqt_date']], values='tot_return_eqt_date', index=['date'],columns=['eqt_code'])
    corr = df.corr()
    corr = corr.fillna(0)
    return corr


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    
    
    
    
    
    
    

