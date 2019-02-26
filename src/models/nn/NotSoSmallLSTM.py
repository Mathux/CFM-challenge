#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:17:21 2019

@author: evrardgarcelon
"""

from keras.layers import (Dense, Dropout, Embedding, PReLU,
                          SpatialDropout1D, concatenate, Flatten, MaxPooling1D,
                          RepeatVector, LSTM, Bidirectional, BatchNormalization)
from keras.models import Model, Input

from src.models.nn.model import GeneralLSTM
from src.models.nn.janet import JANET




class NotSoSmallLSTM(GeneralLSTM):
    def __init__(self,
                 data,
                 eqt_embeddings_size=20,
                 lstm_out_dim=11,
                 use_lstm=True,
                 dropout_rate=0.5,
                 dropout_spatial_rate=0.5,
                 dropout_lstm=0.5,
                 dropout_lstm_rec=0.5,
                 kernel_size=3,
                 loss='binary_crossentropy',
                 optimizer=None):
        super(NotSoSmallLSTM, self).__init__(
            data,
            eqt_embeddings_size=eqt_embeddings_size,
            lstm_out_dim=lstm_out_dim,
            use_lstm=use_lstm,
            dropout_rate=dropout_rate,
            dropout_spatial_rate=dropout_spatial_rate,
            dropout_lstm=dropout_lstm,
            dropout_lstm_rec=dropout_lstm_rec,
            kernel_size=kernel_size,
            loss=loss,
            optimizer=optimizer)

        self.model, self.inputnames = self.create_model()

    def create_model(self):
        eqt_code_input = Input(shape=[1], name='eqt_code_input')
        eqt_emb = Embedding(
            output_dim=self.eqt_embeddings_size,
            input_dim=self.n_eqt,
            input_length=1,
            name='eqt_embeddings')(eqt_code_input)
        eqt_emb = Dropout(self.dropout_spatial_rate)(eqt_emb)
        eqt_emb = Flatten()(eqt_emb)
        

        nb_eqt_traded_input = Input(shape=[1], name='nb_eqt_traded_input')
        nb_eqt_traded_emb = Embedding(
            output_dim=self.eqt_embeddings_size//2,
            input_dim=self.n_eqt,
            input_length=1)(nb_eqt_traded_input)
        nb_eqt_traded = Dropout(self.dropout_spatial_rate)(nb_eqt_traded_emb)
        nb_eqt_traded = Flatten()(nb_eqt_traded)
        
        nb_nan_input = Input(shape=[1], name='nb_nan_input', dtype = 'int64')
        nb_nans_data_emb = Embedding( output_dim=self.eqt_embeddings_size//2,
            input_dim=71,
            input_length=1)(nb_nan_input)
        nb_nans_data = Dropout(self.dropout_spatial_rate)(nb_nans_data_emb)
        nb_nans_data = Flatten()(nb_nans_data)
        
#        nb_days_eqt_traded_input = Input(shape=[1], name='nb_days_eqt_traded_input', dtype = 'int64')
#        nb_days_eqt_traded = Embedding( output_dim=self.eqt_embeddings_size//2,
#            input_dim=1512,
#            input_length=1)(nb_days_eqt_traded_input)
#        nb_days_eqt_traded = Dropout(self.dropout_spatial_rate)(nb_days_eqt_traded)
#        nb_days_eqt_traded = Flatten()(nb_days_eqt_traded)
        
        context_eqt_day = concatenate([eqt_emb,nb_eqt_traded,nb_nans_data])
        context_eqt_day = Dense(128, activation = 'linear')(context_eqt_day)
        context_eqt_day = PReLU()(context_eqt_day)
        context_eqt_day = Dropout(self.dropout_rate)(context_eqt_day)
        context_eqt_day = BatchNormalization()(context_eqt_day)
        
        
        returns_input = Input(
            shape=(self.returns_length, 1), name='returns_input')
        
        market_returns_input = Input(
            shape=(self.returns_length, 1), name='market_returns_input')
        
        return_diff_to_market_input = Input(
                shape=(self.returns_length, 1), 
                name='return_diff_to_market_input')
                
        eqt_avg_returns_input = Input(
                shape=(self.returns_length, 1), name='eqt_avg_returns_input')
        
        log_vol_input = Input(shape=(self.returns_length, 1), name='log_vol_input')
        
        market_log_vol_input = Input(shape=(self.returns_length, 1), name='market_log_vol_input')
        
        log_vol_diff_to_market_input = Input(shape=(self.returns_length, 1), name='log_vol_diff_to_market_input')
                
        eqt_avg_log_vol_input = Input(
                shape=(self.returns_length, 1), name='eqt_avg_log_vol_input')
        
        returns_features = concatenate([market_returns_input,
                                        return_diff_to_market_input,
                                        returns_input,
                                        eqt_avg_returns_input,
                                        market_log_vol_input,
                                        log_vol_diff_to_market_input,
                                        log_vol_input,
                                        eqt_avg_log_vol_input
                                        ])
                
        returns_features_lstm = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = True)(returns_input)
                
        eqt_market_returns_features_lstm = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = True)(returns_features)
        
        eqt_market_returns = concatenate([returns_features_lstm,eqt_market_returns_features_lstm])
        eqt_market_returns = Dense(256,activation = 'linear')(eqt_market_returns)
        eqt_market_returns = PReLU()(eqt_market_returns)
        eqt_market_returns = BatchNormalization()(eqt_market_returns)
        
    
        x = concatenate([returns_features_lstm, eqt_market_returns_features_lstm, context_eqt_day, 
                        ])
        
        x = Dense(256,activation = 'linear')(x)
        
        x = PReLU()(x)
        
        x = Dropout(self.dropout_rate)(x)
        
        x = BatchNormalization()(x)
        
        x = Dense(256,activation = 'linear')(x)
        
        x = PReLU()(x)
        
        x = Dropout(self.dropout_rate)(x)
        
        x = BatchNormalization()(x)
        
        output = Dense(2,activation = 'softmax',name = 'output')(x)

        
        model = Model(
            inputs=[eqt_code_input, 
                    nb_eqt_traded_input, 
                    nb_nan_input,
                    returns_input, 
                    market_returns_input, 
                    return_diff_to_market_input,
                    eqt_avg_returns_input,
                    log_vol_input, 
                    market_log_vol_input, 
                    log_vol_diff_to_market_input,
                    eqt_avg_log_vol_input],
            outputs=[output])

        inputs = ["eqt_code_input", 
                  "nb_eqt_traded", 
                  "nb_nans_data",
                  "returns_input", 
                  "market_returns_input",
                  "return_diff_to_market_input",
                  "eqt_avg_returns", 
                  "log_vol_input", 
                  "market_log_vol_input",
                  "log_vol_diff_to_market_input",
                  "eqt_avg_log_vol",
                  ]
        return model, inputs
    

if __name__ == '__main__':
    from src.tools.experiment import Experiment
    from src.tools.dataloader import Data
    from src.tools.utils import plot_training
    from keras.utils import plot_model
    
    exp = Experiment(modelname="not_small_janet")
    data = Data(small=False, 
                verbose=True, 
                ewma = False, 
                aggregate = False, 
                kfold = 3)
    exp.addconfig("data", data.config)

    model = NotSoSmallLSTM(data, use_lstm=True)
    exp.addconfig("model", model.config)

    plot_model(model.model, to_file=exp.pnggraph, show_shapes=True)

    model.model.summary()
    # Fit the model
    history = model.compile_fit(
        checkpointname=exp.modelname, 
        epochs=50, 
        plateau_patience=5, 
        verbose=1, 
        kfold = 3, 
        batch_size = 4092)

    exp.addconfig("learning", model.learning_config)
    exp.saveconfig(verbose=True)

    plot_training(
        history, show=False, losspath=exp.pngloss, accpath=exp.pngacc)

    # model.create_submission(exp.allpath("predictions.csv"),'end_of_day_return'
    # ,'end_of_day_return')

#%%