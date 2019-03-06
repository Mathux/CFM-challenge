#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:17:21 2019

@author: evrardgarcelon
"""

from keras.layers import (Dense, Dropout, Embedding, PReLU, SpatialDropout1D,
                          concatenate, Flatten, MaxPooling1D, RepeatVector,
                          LSTM, Bidirectional, BatchNormalization)
from keras.models import Model, Input


from src.models.nn.model import GeneralLSTM
from src.models.nn.janet import JANET


class NotSoSmallLSTM(GeneralLSTM):
    def __init__(self,
                 data,
                 eqt_embeddings_size=50,
                 lstm_out_dim=35,
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
        

        nb_eqt_traded = Input(shape=[1], name='nb_eqt_traded_input')
        #nb_eqt_traded_emb = Embedding(
        #    output_dim=self.eqt_embeddings_size//2,
        #    input_dim=self.n_eqt,
        #    input_length=1)(nb_eqt_traded_input)
        #nb_eqt_traded = Dropout(self.dropout_spatial_rate)(nb_eqt_traded_emb)
        #nb_eqt_traded = Flatten()(nb_eqt_traded)
        
        nb_nans_data = Input(shape=[1], name='nb_nan_input')
        #nb_nans_data_emb = Embedding( output_dim=self.eqt_embeddings_size//2,
        #    input_dim=71,
        #    input_length=1)(nb_nan_input)
        #nb_nans_data = Dropout(self.dropout_spatial_rate)(nb_nans_data_emb)
        #nb_nans_data = Flatten()(nb_nans_data)
        
        nb_days_eqt_traded = Input(shape=[1], name='nb_days_eqt_traded_input')
        #nb_days_eqt_traded = Embedding( output_dim=self.eqt_embeddings_size//2,
        #    input_dim=1512,
        #    input_length=1)(nb_days_eqt_traded_input)
        #nb_days_eqt_traded = Dropout(self.dropout_spatial_rate)(nb_days_eqt_traded)
        #nb_days_eqt_traded = Flatten()(nb_days_eqt_traded)
        
        context_eqt_day = concatenate([eqt_emb,nb_eqt_traded,nb_nans_data,nb_days_eqt_traded])
        context_eqt_day = Dense(32, activation = 'linear')(context_eqt_day)
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
        
#        log_vol_input = Input(shape=(self.returns_length, 1), name='log_vol_input')
        
#        market_log_vol_input = Input(shape=(self.returns_length, 1), name='market_log_vol_input')
        
#        returns_features = concatenate([market_returns_input,
#                                        return_diff_to_market_input,
#                                        returns_input,
#                                        eqt_avg_returns_input,
#                                        market_log_vol_input,
#                                        log_vol_input,
#                                        ])
#        
        return_diff_to_market_features = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = False)(return_diff_to_market_input)

        market_returns_features = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = False)(market_returns_input)
        
        eqt_avg_returns_features = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = False)(eqt_avg_returns_input)
        
#        log_vol_features = JANET(
#            self.lstm_out_dim,
#            return_sequences=False,
#            dropout=self.dropout_lstm,
#            recurrent_dropout=self.dropout_lstm_rec, unroll = False)(log_vol_input)
#        
#        market_log_vol_features = JANET(
#            self.lstm_out_dim,
#            return_sequences=False,
#            dropout=self.dropout_lstm,
#            recurrent_dropout=self.dropout_lstm_rec, unroll = False)(market_log_vol_input)
#                
        returns_features = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = False)(returns_input)
                
#        vol_features = concatenate([log_vol_features,market_log_vol_features])
        
        market_features = concatenate([returns_features,eqt_avg_returns_features,market_returns_features,return_diff_to_market_features])
        
#        vol_features = Dense(self.lstm_out_dim, activation = 'linear')(vol_features)
#        vol_features = PReLU()(vol_features)
#        vol_features = Dropout(self.dropout_rate)(vol_features)
#        vol_features = BatchNormalization()(vol_features)
#        
        return_features = Dense(self.lstm_out_dim,activation = 'linear')(returns_features)
        return_features = PReLU()(return_features)
        return_features = Dropout(self.dropout_rate)(return_features)
        return_features = BatchNormalization()(return_features)
        
    
        market_features = concatenate([return_features, market_features])
        market_features = Dense(128,activation = 'linear')(market_features)
        market_features = PReLU()(market_features)
        market_features = BatchNormalization()(market_features)
        
        handmade_features = Input(shape = (len(self.non_return_cols)-3,), name = 'handmade_features')
        
        x = concatenate([market_features, return_features, context_eqt_day, 
                        handmade_features])
        
        x = Dense(128,activation = 'linear')(x)
        
        x = PReLU()(x)
        
        x = Dropout(self.dropout_rate)(x)
        
        x = BatchNormalization()(x)
#        
#        x = Dense(128,activation = 'linear')(x)
#        
#        x = PReLU()(x)
#        
#        x = Dropout(self.dropout_rate)(x)
#        
#        x = BatchNormalization()(x)
        
        output = Dense(2,activation = 'softmax',name = 'output')(x)

        
        model = Model(
            inputs=[eqt_code_input, 
                    nb_eqt_traded, 
                    nb_nans_data,
                    nb_days_eqt_traded,
                    returns_input, 
                    market_returns_input, 
                    return_diff_to_market_input,
                    eqt_avg_returns_input,
                    handmade_features],
            outputs=[output])

        inputs = ["eqt_code_input", 
                  "nb_eqt_traded", 
                  "nb_nans_data",
                  'nb_days_eqt_traded',
                  "returns_input", 
                  "market_returns_input",
                  "return_diff_to_market_input",
                  "eqt_avg_returns", 
                  "handmade_features_input"
                  ]
        return model, inputs
    


if __name__ == '__main__':
    from src.tools.experiment import Experiment
    from src.tools.dataloader import Data
    from src.tools.utils import plot_training

    KFOLDS = 0
    EPOCHS = 1
    
    exp = Experiment(modelname="not_small_janet")
    data = Data(
        small=False, verbose=True, ewma=False, aggregate=False)

    exp.addconfig("data", data.config)

    model = NotSoSmallLSTM(data, use_lstm=True)
    exp.addconfig("model", model.config)
#    from keras.utils import plot_model
#    plot_model(model.model, to_file=exp.pnggraph, show_shapes=True)

    model.model.summary()
    # Fit the model
    histories = model.compile_fit(
        checkpointname=exp.modelname,
        epochs=EPOCHS,
        plateau_patience=5,
        stop_patience=10,
        verbose=1,
        batch_size=8500,
        best = True,
        )

    exp.addconfig("learning", model.learning_config)
    exp.saveconfig(verbose=True)

    for el, history in enumerate(histories):
        plot_training(
            history,
            show=False,
            losspath=exp._pngloss(el + 1),
            accpath=exp._pngacc(el + 1))

    model.create_submission(
        exp.modelname,
        bincsv=exp.allpath("predictions_bin.csv"),
        probacsv=exp.allpath("predictions_proba.csv"))
