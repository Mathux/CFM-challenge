#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:17:21 2019

@author: evrardgarcelon
"""

from keras.layers import (Dense, Dropout, Embedding, PReLU, SpatialDropout1D,
                          concatenate, Flatten, MaxPooling1D, RepeatVector,
                          LSTM, Bidirectional, BatchNormalization, Reshape)
from keras.models import Model, Input
import keras


from src.models.nn.model import GeneralLSTM
from src.models.nn.janet import JANET


class NotSoSmallLSTM(GeneralLSTM):
    def __init__(self,
                 data,
                 eqt_embeddings_size=20,
                 lstm_out_dim=50,
                 dropout_rate=0.5,
                 dropout_spatial_rate=0.5,
                 dropout_lstm=0.5,
                 dropout_lstm_rec=0.5,
                 loss='binary_crossentropy',
                 optimizer=None):
        super(NotSoSmallLSTM, self).__init__(
            data,
            eqt_embeddings_size=eqt_embeddings_size,
            lstm_out_dim=lstm_out_dim,
            use_lstm=True,
            dropout_rate=dropout_rate,
            dropout_spatial_rate=dropout_spatial_rate,
            dropout_lstm=dropout_lstm,
            dropout_lstm_rec=dropout_lstm_rec,
            loss=loss,
            optimizer=optimizer)

        self.model, self.inputnames = self.create_model()

    def create_model(self):
        
         
        ### Temporal informations
        returns_input = Input(shape=(self.returns_length, 1), name='returns_input')
        
        return_features =  JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = False,
            kernel_initializer='random_uniform')(returns_input)
        
        return_features = Dense(32,activation = 'linear')(return_features)
        return_features = PReLU()(return_features)
        return_features = Dropout(self.dropout_rate)(return_features)
        return_features = BatchNormalization()(return_features)
        
        
        ###Handmade Features input
        handmade_features_input = Input(shape = (8,), 
                                  name = 'handmade_features')
        handmade_features = Dense(32, activation = 'linear')(handmade_features_input)
        handmade_features = PReLU()(handmade_features)
        handmade_features = Dropout(self.dropout_rate)(handmade_features)
        handmade_features = BatchNormalization()(handmade_features)
        
        ### Final Concatenation
        x = concatenate([return_features,handmade_features_input])
        
        x = Dense(32,activation = 'linear')(x)
        
        x = PReLU()(x)
        
        x = Dropout(self.dropout_rate)(x)
        
        x = BatchNormalization()(x)
        
        x = Dense(32,activation = 'linear')(x)
        
        x = PReLU()(x)

        x = Dropout(self.dropout_rate)(x)
        
        x = BatchNormalization()(x)
        
        output = Dense(2,activation = 'softmax',name = 'output')(x)

        
        model = Model(
            inputs =[
                    returns_input,
                    handmade_features_input],
            outputs=[output])

        inputs = ["returns_input",
                  "handmade_features_input"
                  ]
        return model, inputs
    


if __name__ == '__main__':
    from src.tools.experiment import Experiment
    from src.tools.dataloader import Data
    from src.tools.utils import plot_training

    KFOLDS = 0
    EPOCHS = 200
    
    exp = Experiment(modelname="not_small_janet")
    data = Data(
        small=False, verbose=True, ewma=False, aggregate=False)

    exp.addconfig("data", data.config)

    model = NotSoSmallLSTM(data)
    exp.addconfig("model", model.config)
   # from keras.utils import plot_model
   # plot_model(model.model, to_file=exp.pnggraph, show_shapes=True)

    model.model.summary()
    # Fit the model
    histories = model.compile_fit(
        checkpointname=exp.modelname,
        epochs=EPOCHS,
        plateau_patience=5,
        stop_patience=15,
        verbose=1,
        batch_size=8192,
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
    
