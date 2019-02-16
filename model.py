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
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

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
    
    def compile_fit(self,epochs = 30, batch_size = 64, verbose = 0) :
        
        model = self.create_model()
        
        if self.optimizers is None :
            opti = Nadam()
        else :
            opti = self.optimizers
        
        early_stop = EarlyStopping(monitor='val_acc', patience=3, 
                                   verbose=verbose, 
                                   min_delta=1e-8, 
                                   mode='min')
        checkpointer = ModelCheckpoint(filepath="test.hdf5", 
                                       verbose=verbose, 
                                       save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', 
                                      factor=0.9,
                                      patience=5, 
                                      min_lr=0.000001, 
                                      verbose=verbose)
        model.compile(optimizer = opti, loss = self.loss, metrics = ['acc'])
        X_train,y_train,X_val,y_val = self.process_data(X,y)
        history = model.fit(X_train, y_train, epochs = epochs, 
                            batch_size = batch_size, 
                            verbose=verbose, 
                            validation_data=(X_val, y_val),
                            callbacks=[reduce_lr, checkpointer,early_stop],
                            shuffle=True)

        return history
    
if __name__ == '__main__' :
    
    data = Data(small = True)
    X,y = data.train.data, data.train.labels
    model = LSTMModel(X,y, use_lstm = True)
    model.create_model().summary()
        
        
        

