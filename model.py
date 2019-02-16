#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:33:23 2019

@author: evrardgarcelon
"""

import numpy as np

import keras
from keras.models import Model, Input
from keras.layers import (Dense, Dropout, Embedding, Conv1D, PReLU, 
SpatialDropout1D, concatenate, BatchNormalization, Flatten, LSTM)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam
from keras.utils import to_categorical

from processing_data import Data

from sklearn.preprocessing import LabelEncoder


class LSTMModel(object) :
    
    def __init__(self,X, y, eqt_embeddings_size = 80, lstm_out_dim = 5, 
                 use_lstm = False, dropout_rate = 0.1, kernel_size = 2, loss =
                 'binary_crossentropy', optimizer = None) :
        self.X = X
        self.y = y
        self.eqt_embeddings_size = eqt_embeddings_size
        self.n_eqt = X['eqt_code'].unique()
        self.return_cols = [c for c in X.columns if c.endswith(':00')]
        self.non_return_cols = [c for c in X.columns if not c.endswith(':00')]
        self.returns_length = len(self.return_cols)
        self.lstm_out_dim = lstm_out_dim
        self.use_lstm = use_lstm
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.loss = loss
        self.optimizer = optimizer
        
    def create_model(self):
        
        #First create the eqt embeddings 
        eqt_code_input = Input(shape =(1,), dtype = 'int64', 
                               name = 'eqt_code_input')
        eqt_emb = Embedding(output_dim = self.eqt_embeddings_size, 
                            input_dim = 1,
                            name = 'eqt_embeddings')(eqt_code_input)
        eqt_emb = SpatialDropout1D(self.dropout_rate)(eqt_emb)
        eqt_emb = Flatten()(eqt_emb)
        
        # Then the LSTM/CNN1D for the returns time series 
        returns_input = Input(shape = (self.returns_length,1),
                              name = 'returns_input')
        
        if self.use_lstm :
            returns_lstm = LSTM(self.lstm_out_dim)(returns_input)
        else :
            returns_lstm = Conv1D(filters = self.lstm_out_dim, 
                                  kernel_size = self.kernel_size, 
                                  activation = 'linear',
                                  name = 'returns_conv')(returns_input)
            returns_lstm = Flatten()(returns_lstm)
        returns_lstm = PReLU()(returns_lstm)
        returns_lstm = Dropout(self.dropout_rate)(returns_lstm)
        
        # and the the LSTM/CNN part for the volatility time series
        
        vol_input = Input(shape = (self.returns_length,1),name = 'vol_input')
        
        if self.use_lstm :
            vol_lstm = LSTM(self.lstm_out_dim)(vol_input)
        else : 
            vol_lstm = Conv1D(filters = self.lstm_out_dim, 
                              kernel_size = self.kernel_size, 
                              activation = 'linear',
                              name = 'returns_conv')(vol_input)
            vol_lstm = Flatten()(vol_lstm)
        vol_lstm = PReLU()(vol_lstm)
        vol_lstm = Dropout(self.dropout_rate)(vol_lstm)
        
        x = concatenate([eqt_emb,returns_lstm,vol_lstm])
        x = Dense(32,activation = 'linear')(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32,activation = 'linear')(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Finally concatenate the handmade features and the one from 
        # the embeddings from the lstms/convultions
        
        handmade_features_input = Input(shape =(len(self.non_return_cols),),  
                                        name = 'handmade_features_input')
        
        x = concatenate([handmade_features_input,x])
        x = Dense(32,activation = 'linear')(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32,activation = 'linear')(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        output = Dense(1,activation = 'sigmoid')(x)
        
        model = Model(inputs=[eqt_code_input, returns_input, vol_input,
                              handmade_features_input], outputs=[output])
        return model
        
        
        
        
        
        
    def process_data(self,X,y) : 
        pass
    
if __name__ == '__main__' :
    
    data = Data(small = True)
    X,y = data.train.data, data.train.labels
    model= LSTMModel(X,y, use_lstm = True)
        
        
        

