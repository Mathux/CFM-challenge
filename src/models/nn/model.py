#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:33:23 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
import pandas as pd
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from src.models.nn.CLR import CyclicLR


class GeneralModel:
    def __init__(self):
        return

    def process_data(self, data, labels=None):
        def create_input(name, data, eps=10**-10):
            if name == "eqt_code_input":
                temp_eqt = LabelEncoder()
                temp_eqt.fit(data['eqt_code'].values)
                return temp_eqt.transform(data['eqt_code'].values)
            
            elif name == 'date' :
                temp_name = LabelEncoder()
                temp_name.fit(data['date'].values)
                return temp_name.transform(data['date'].values)
            

            elif name == "nb_eqt_traded":
#                scaler = MinMaxScaler()
                scaler = LabelEncoder()
                temp_d = data['countd_product'].values
                scaler.fit(temp_d)
                temp_d = scaler.transform(data['countd_product'].values)
                return temp_d

            elif name == "nb_nans_data":
#                scaler = MinMaxScaler()
                scaler = LabelEncoder()
                temp_d = data['return_nan'].values.astype('int64')
                temp_d = scaler.fit_transform(temp_d)
                temp_d = temp_d.reshape((len(temp_d),1))
                return temp_d

            elif name == 'nb_days_eqt_traded':
                scaler = MinMaxScaler()
                scaler = LabelEncoder()
                temp_d = data['countd_date'].values.astype('int64')
                temp_d = scaler.fit_transform(temp_d)
                temp_d = temp_d.reshape((len(temp_d),1))
                return temp_d

            elif name == "returns_input":
                temp_returns = data[self.return_cols].values
                temp_returns = self.scale(temp_returns)
                temp_returns = temp_returns.reshape((temp_returns.shape[0],
                                                     temp_returns.shape[1], 1))
                return temp_returns

            elif name == "market_returns_input":
                temp_market_returns = data.groupby('date')[
                    self.return_cols].mean()
                data.set_index(['date'], inplace=True)
                self.market_return_cols = [
                    'market_returns_' + c for c in self.return_cols
                ]
                data[self.market_return_cols] = temp_market_returns
                data.reset_index(inplace=True)
                temp_market_returns = data[self.market_return_cols].values
                temp_market_returns = self.scale(temp_market_returns)
                temp_market_returns = temp_market_returns.reshape(
                    (temp_market_returns.shape[0],
                     temp_market_returns.shape[1], 1))
                
                data = data.drop(self.market_return_cols, axis=1)
                return temp_market_returns

            elif name == 'eqt_avg_returns':
                temp_eqt_returns = data.groupby('eqt_code')[
                    self.return_cols].mean()
                data.set_index(['eqt_code'], inplace=True)
                self.eqt_return_cols = [
                    'eqt_returns_' + c for c in self.return_cols
                ]
                data[self.eqt_return_cols] = temp_eqt_returns
                data.reset_index(inplace=True)
                temp_eqt_returns = data[self.eqt_return_cols].values
                temp_eqt_returns = self.scale(temp_eqt_returns)
                temp_eqt_returns = temp_eqt_returns.reshape(
                    (temp_eqt_returns.shape[0], temp_eqt_returns.shape[1], 1))
                data = data.drop(self.eqt_return_cols, axis=1)
                return temp_eqt_returns
            
            elif name == "handmade_features_input":
                temp_non_return_cols = [col for col in self.non_return_cols if not col in ["kurt_return_date_eqt", 'skew_return_date_eqt','max_drawdown_return_date_eqt']]
                return self.scale(data[temp_non_return_cols].values)
            
            elif name == 'rolling_ewma_returns' :
                temp_returns = data[self.return_cols].ewm(halflife = 1.5).mean().values
                temp_returns = temp_returns.reshape((temp_returns.shape[0],
                                                     temp_returns.shape[1], 1))
                return temp_returns
            
            elif name == 'rolling_var_returns' : 
                temp_returns = data.groupby('eqt_code')[self.return_cols]
                temp_var = temp_returns.std().rolling(11, min_periods= 1, axis = 1).sum()
                data.set_index(['eqt_code'], inplace=True)
                self.var_return_cols = [
                    'var_' + c for c in self.return_cols
                ]
                data[self.var_return_cols] = temp_var
                data.reset_index(inplace=True)
                temp_var = data[self.var_return_cols].values
                temp_var = temp_var.reshape(
                    (temp_var.shape[0], temp_var.shape[1], 1))
                return temp_var
                
                
        input_data = [create_input(name, data) for name in self.inputnames]
        

        if labels is not None:
            labels = to_categorical(
                labels.drop('ID', axis=1)['end_of_day_return'].values)
            return input_data, labels
        else:
            return input_data
    
    def scale(self,X) :
        return StandardScaler().fit_transform(X)

    def compile_fit(self,
                    checkpointname,
                    epochs=50,
                    plateau_patience=15,
                    stop_patience=15,
                    batch_size=8192,
                    verbose=0,
                    kfold=None,
                    best = False):

        conf = {}

        if self.optimizer is None:
            conf["optimizer"] = {
                "lr": 0.001,
                "rho": 0.9,
                "epsilon": None,
                "decay": 10**-7,
                "name": "RMSprop",
            }

            opti = RMSprop(
                lr=conf["optimizer"]["lr"],
                rho=conf["optimizer"]["rho"],
                epsilon=conf["optimizer"]["epsilon"],
                decay=conf["optimizer"]["decay"])
        else:
            # add config
            opti = self.optimizer

        conf["EarlyStopping"] = {"monitor": "val_loss", "patience": stop_patience}
        

        def callbacks_intrain():
            early_stop = EarlyStopping(
                monitor=conf["EarlyStopping"]['monitor'],
                patience=conf["EarlyStopping"]["patience"],
                verbose=verbose)
            
            checkpointer = ModelCheckpoint(
                filepath=checkpointname,
                verbose=verbose,
                save_best_only=conf["ModelCheckpoint"]["save_best_only"],
                save_weights_only=conf["ModelCheckpoint"]["save_weights_only"])

            
            reduce_lr = ReduceLROnPlateau(
                monitor=conf["ReduceLROnPlateau"]["monitor"],
                factor=conf["ReduceLROnPlateau"]["factor"],
                patience=conf["ReduceLROnPlateau"]["patience"],
                min_lr=conf["ReduceLROnPlateau"]["min_lr"],
                verbose=verbose)
            
            base_lr = conf["optimizer"]["lr"]
            clr  = CyclicLR(mode='exp_range', 
                            max_lr=base_lr*6, 
                            base_lr=base_lr, 
                            step_size=100)

            return early_stop, checkpointer, reduce_lr, clr
        
        conf["ModelCheckpoint"] = {
            "save_best_only": True,
            "save_weights_only": True
        }

        conf["ReduceLROnPlateau"] = {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": plateau_patience,
            "min_lr": 10**-6
        }

        conf["metrics"] = ["acc"]

        early_stop, checkpointer, reduce_lr, clr = callbacks_intrain()
                
        self.model.compile(
            optimizer=opti,
            loss={'output': self.loss},
            metrics={'output': conf["metrics"]},
            loss_weights=[1])

        conf["epochs"] = epochs
        conf["batch_size"] = batch_size

        if kfold is None:
            X_train, y_train = self.process_data(self.data.train.data,
                                                 self.data.train.labels)
            X_val, y_val = self.process_data(self.data.val.data,
                                             self.data.val.labels)

            history = self.model.fit(
                X_train,
                y_train,
                epochs=conf["epochs"],
                batch_size=conf["batch_size"],
                verbose=verbose,
                validation_data=(X_val, y_val),
                callbacks=[checkpointer, early_stop, reduce_lr])
        else:
            history = []
            for k in range(kfold):
                if verbose:
                    print("Training on the fold ", k)
                X_train, y_train = self.data.merge_folds(k)
                X_train, y_train = self.process_data(X_train,
                                                     y_train)
                
                X_val, y_val = self.process_data(self.data.folds[k].data,
                                                 self.data.folds[k].labels)

                opti = RMSprop(
                    lr=conf["optimizer"]["lr"],
                    rho=conf["optimizer"]["rho"],
                    epsilon=conf["optimizer"]["epsilon"],
                    decay=conf["optimizer"]["decay"])
                            
                self.model.compile(
                    optimizer=opti,
                    loss={'output': self.loss},
                    metrics={'output': conf["metrics"]},
                    loss_weights=[1])
                
                early_stop, checkpointer, reduce_lr, clr = callbacks_intrain()
                
                hist = self.model.fit(
                    X_train,
                    y_train,
                    epochs=conf["epochs"],
                    batch_size=conf["batch_size"],
                    verbose=verbose,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpointer, early_stop, reduce_lr])
                
                history.append(hist)
                # Load the best model
                
                if best :
                    self.model.load_weights(checkpointname)
                
        self.learning_config = conf
        
        if isinstance(history,keras.callbacks.History) :
            history = [history]
        
        return history

    def create_submission(self, modelname, bincsv, probacsv):
        from src.tools.utils import submission
        # Load the best model
        self.model.load_weights(modelname)
        X_test = self.process_data(self.data.test.data)
        predictions = self.model.predict(X_test)
        bin_pred = np.argmax(predictions, axis=1)
        proba_pred = predictions[:, 1]

        submission(bin_pred, ID=self.data.test.data["ID"], name=bincsv)
        submission(proba_pred, ID=self.data.test.data["ID"], name=probacsv)


class GeneralLSTM(GeneralModel):
    def __init__(self,
                 data,
                 eqt_embeddings_size=80,
                 lstm_out_dim=256,
                 use_lstm=True,
                 dropout_rate=0.5,
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
