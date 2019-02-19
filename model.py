#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:33:23 2019

@author: evrardgarcelon, mathispetrovich
"""

import utils
import numpy as np
import pylab as plt

from janet import JANET

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import (Dense, Dropout, Embedding, Conv1D, PReLU,
                          SpatialDropout1D, concatenate, BatchNormalization,
                          Flatten, LSTM, MaxPooling1D)
from keras.models import Model, Input
from keras.optimizers import Nadam, RMSprop
from keras.utils import plot_model, to_categorical
from sklearn.preprocessing import LabelEncoder
from processing_data import Data


class LSTMModel(object):
    def __init__(self,
                 data,
                 eqt_embeddings_size=80,
                 lstm_out_dim=64,
                 use_lstm=True,
                 dropout_rate=0.8,
                 dropout_spatial_rate=0.5,
                 dropout_lstm=0.3,
                 dropout_lstm_rec=0.3,
                 kernel_size=3,
                 loss='binary_crossentropy',
                 optimizer=None):
        self.data = data
        self.eqt_embeddings_size = eqt_embeddings_size
        self.n_eqt = self.data.nunique
        self.return_cols = [
            c for c in data.train.data.columns if c.endswith(':00')
        ]
        self.non_return_cols = [
            c for c in data.train.data.columns if (not c.endswith(':00')) and (
                c not in
                ['ID', 'eqt_code', 'date', 'Unnamed: 0_x', 'Unnamed: 0_y'])
        ]
        self.returns_length = len(self.return_cols)
        self.lstm_out_dim = lstm_out_dim
        self.use_lstm = use_lstm
        
        self.dropout_rate = dropout_rate
        self.dropout_spatial_rate = dropout_spatial_rate
        
        self.dropout_lstm = dropout_lstm
        self.dropout_lstm_rec = dropout_lstm_rec
        
        self.kernel_size = kernel_size
        self.loss = loss
        self.optimizer = optimizer
        self.model = self.create_model()

        self.config = {
            "eqt_embeddings_size": self.eqt_embeddings_size,
            "lstm_out_dim": self.lstm_out_dim,
            "use_lstm": self.use_lstm,
            "dropout_rate": self.dropout_rate,
            "dropout_spatial_rate": self.dropout_spatial_rate,
            "droupout_lstm": self.dropout_lstm,
            "droupout_lstm_rec": self.dropout_lstm_rec,
            "kernel_size": self.kernel_size,
            "loss": self.loss
        }

    def create_model(self):
        # First create the eqt embeddings
        eqt_code_input = Input(shape=[1], name='eqt_code_input')
        eqt_emb = Embedding(
            output_dim=self.eqt_embeddings_size,
            input_dim=self.n_eqt + 1,
            input_length=1,
            name='eqt_embeddings')(eqt_code_input)
        eqt_emb = SpatialDropout1D(self.dropout_spatial_rate)(eqt_emb)
        eqt_emb = Flatten()(eqt_emb)

        # Then the LSTM/CNN1D for the returns time series
        returns_input = Input(
            shape=(self.returns_length, 1), name='returns_input')

        if self.use_lstm:
            returns_lstm = JANET(self.lstm_out_dim,
                                 return_sequences = False,
                                 dropout = 0.3,
                                 recurrent_dropout = 0.3)(returns_input)
        else:
            returns_lstm = Conv1D(
                filters=self.lstm_out_dim,
                kernel_size=self.kernel_size,
                activation='linear',
                name='returns_conv')(returns_input)
            returns_lstm = Dropout(self.dropout_rate)(returns_lstm)
            returns_lstm = MaxPooling1D()(returns_lstm)
            returns_lstm = Flatten()(returns_lstm)
        returns_lstm = Dense(32, activation='linear')(returns_lstm)
        returns_lstm = Dropout(self.dropout_rate)(returns_lstm)
        returns_lstm = PReLU()(returns_lstm)

        # and the the LSTM/CNN part for the volatility time series
        vol_input = Input(shape=(self.returns_length, 1), name='vol_input')

        if self.use_lstm:
            vol_lstm = JANET(
                self.lstm_out_dim, return_sequences=False)(vol_input)
        else:
            vol_lstm = Conv1D(
                filters=self.lstm_out_dim,
                kernel_size=self.kernel_size,
                activation='linear',
                name='vol_conv')(vol_input)
            vol_lstm = Dropout(self.dropout_rate)(vol_lstm)
            vol_lstm = MaxPooling1D()(vol_lstm)
            vol_lstm = Flatten()(vol_lstm)
        vol_lstm = Dense(32, activation='linear')(vol_lstm)
        vol_lstm = Dropout(self.dropout_rate)(vol_lstm)
        vol_lstm = PReLU()(vol_lstm)

        x = concatenate([eqt_emb, returns_lstm, vol_lstm])
        x = Dense(128, activation='linear')(x)
        x = PReLU()(x)

        # Finally concatenate the handmade features and the one from
        # the embeddings from the lstms/convultions

        handmade_features_input = Input(
            shape=(len(self.non_return_cols), ),
            name='handmade_features_input')

        x = concatenate([handmade_features_input, x])
        x = Dense(128, activation='linear')(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='linear')(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        output = Dense(2, activation='softmax')(x)

        model = Model(
            inputs=[
                eqt_code_input, returns_input, vol_input,
                handmade_features_input
            ],
            outputs=[output])
        return model

    def process_data(self, data, labels=None):
        input_data = []

        temp_eqt = LabelEncoder()
        temp_eqt.fit(data['eqt_code'].values)

        input_data.append(temp_eqt.transform(data['eqt_code'].values))

        temp_returns = data[self.return_cols].values
        temp_returns = temp_returns.reshape((temp_returns.shape[0],
                                             temp_returns.shape[1], 1))
        input_data.append(temp_returns)

        temp_vol = np.abs(data[self.return_cols].values)
        temp_vol = temp_vol.reshape((temp_vol.shape[0], temp_vol.shape[1], 1))
        input_data.append(temp_vol)

        input_data.append(data[self.non_return_cols].values)
        del temp_returns, temp_vol

        if labels is not None:
            labels = to_categorical(
                labels.drop('ID', axis=1)['end_of_day_return'].values)

            return input_data, labels
        else:
            return input_data

    def compile_fit(self,
                    checkpointname,
                    epochs=30,
                    plateau_patience=20,
                    batch_size=128,
                    verbose=0):

        conf = {}

        if self.optimizer is None:
            # opt.__str__().split("optimizers.")[1].split(" ")[0]
            # opti = Nadam()
            conf["optimizer"] = {
                "lr": 0.001,
                "rho": 0.9,
                "epsilon": None,
                "decay": 10**-6,
                "name": "RMSprop"
            }

            opti = RMSprop(
                lr=conf["optimizer"]["lr"],
                rho=conf["optimizer"]["rho"],
                epsilon=conf["optimizer"]["epsilon"],
                decay=conf["optimizer"]["decay"])
        else:
            # add config
            opti = self.optimizer

        conf["EarlyStopping"] = {
            "monitor": "val_loss",
            "patience": epochs // 3
        }

        early_stop = EarlyStopping(
            monitor=conf["EarlyStopping"]['monitor'],
            patience=conf["EarlyStopping"]["patience"],
            verbose=verbose)

        conf["ModelCheckpoint"] = {
            "save_best_only": True,
            "save_weights_only": True
        }
        checkpointer = ModelCheckpoint(
            filepath=checkpointname,
            verbose=verbose,
            save_best_only=conf["ModelCheckpoint"]["save_best_only"],
            save_weights_only=conf["ModelCheckpoint"]["save_weights_only"])

        conf["ReduceLROnPlateau"] = {
            "monitor": "val_loss",
            "factor": 0.99,
            "patience": plateau_patience,
            "min_lr": 0.000001
        }

        reduce_lr = ReduceLROnPlateau(
            monitor=conf["ReduceLROnPlateau"]["monitor"],
            factor=conf["ReduceLROnPlateau"]["factor"],
            patience=conf["ReduceLROnPlateau"]["patience"],
            min_lr=conf["ReduceLROnPlateau"]["min_lr"],
            verbose=verbose)

        conf["metrics"] = ["acc"]
        self.model.compile(
            optimizer=opti, loss=self.loss, metrics=conf["metrics"])

        X_train, y_train = self.process_data(self.data.train.data,
                                             self.data.train.labels)
        X_val, y_val = self.process_data(self.data.val.data,
                                         self.data.val.labels)

        conf["epochs"] = epochs
        conf["batch_size"] = batch_size

        history = self.model.fit(
            X_train,
            y_train,
            epochs=conf["epochs"],
            batch_size=conf["batch_size"],
            verbose=verbose,
            validation_data=(X_val, y_val),
            callbacks=[checkpointer, early_stop, reduce_lr])

        self.learning_config = conf
        return history

    def predict_test(self, bincsv, probacsv):
        from utils import submission
        X_test = self.process_data(self.data.test.data)
        predictions = self.model.predict(X_test)
        bin_pred = np.argmax(predictions, axis=1)
        proba_pred = predictions[:, 1]

        submission(bin_pred, ID=self.data.test.data["ID"], name=bincsv)
        submission(proba_pred, ID=self.data.test.data["ID"], name=probacsv)


def plot_training(history, show=True, losspath=None, accpath=None):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    if losspath:
        plt.savefig(losspath)
    if show:
        plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    if accpath:
        plt.savefig(accpath)

    if show:
        plt.show()


if __name__ == '__main__':
    from experiment import Experiment
    exp = Experiment(modelname="janet_droupout")

    data = Data(verbose=True)
    exp.addconfig("data", data.config)

    model = LSTMModel(data, use_lstm=True)
    exp.addconfig("model", model.config)

    plot_model(model.model, to_file=exp.pnggraph, show_shapes=True)

    # Fit the model
    history = model.compile_fit(
        checkpointname=exp.modelname,
        epochs=100,
        plateau_patience=10,
        verbose=1)

    exp.addconfig("learning", model.learning_config)
    exp.saveconfig(verbose=True)

    plot_training(
        history, show=False, losspath=exp.pngloss, accpath=exp.pngacc)

    # Predict on the test dataset
    model.predict_test(bincsv=exp.allpath("predictions_bin.csv"),
                       probacsv=exp.allpath("predictions_proba.csv"))
