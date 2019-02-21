#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:33:23 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop
from keras.utils import plot_model, to_categorical
from sklearn.preprocessing import LabelEncoder

from src.tools.dataloader import Data

class GeneralModel:
    def __init__(self):
        return

    def process_data(self, data, labels=None):
        def create_input(name):
            if name == "eqt_code_input":
                temp_eqt = LabelEncoder()
                temp_eqt.fit(data['eqt_code'].values)
                return temp_eqt.transform(data['eqt_code'].values)

            elif name == "returns_input":
                temp_returns = data[self.return_cols].values
                temp_returns = temp_returns.reshape((temp_returns.shape[0],
                                                     temp_returns.shape[1], 1))
                return temp_returns

            elif name == "vol_input":
                temp_vol = np.abs(data[self.return_cols].values)
                temp_vol = temp_vol.reshape((temp_vol.shape[0],
                                             temp_vol.shape[1], 1))
                return temp_vol

            elif name == "handmade_features_input":
                return data[self.non_return_cols].values

        input_data = [create_input(name) for name in self.inputnames]

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

    def create_submission(self, modelname, bincsv, probacsv):
        from utils import submission
        # Load the best model
        self.model.load_weights(modelname)
        X_test = self.process_data(self.data.test.data, self.dataconfig)
        predictions = self.model.predict(X_test)
        bin_pred = np.argmax(predictions, axis=1)
        proba_pred = predictions[:, 1]

        submission(bin_pred, ID=self.data.test.data["ID"], name=bincsv)
        submission(proba_pred, ID=self.data.test.data["ID"], name=probacsv)


class GeneralLSTM(GeneralModel):
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

        super(GeneralLSTM, self).__init__()

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
