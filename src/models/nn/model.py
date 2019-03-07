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

            elif name == "nb_eqt_traded":
                scaler = MinMaxScaler()
                temp_d = data['countd_product'].values
                temp_d = temp_d.reshape((len(temp_d),1)).astype('float64')
                return scaler.fit_transform(temp_d)

            elif name == "nb_nans_data":
                scaler = MinMaxScaler()
                temp_d = data['return_nan'].values
                temp_d = temp_d.reshape((len(temp_d),1)).astype('float64')
                return scaler.fit_transform(temp_d)

            elif name == 'nb_days_eqt_traded':
                scaler = MinMaxScaler()
                temp_d = data['countd_date'].values
                temp_d = temp_d.reshape((len(temp_d),1)).astype('float64')
                return scaler.fit_transform(temp_d)

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

            elif name == 'market_kurt_returns':
                temp_kurt_returns = data.groupby('date')[
                    self.return_cols].apply(pd.DataFrame.kurt)
                data.set_index(['date'], inplace=True)
                self.kurt_return_cols = [
                    'kurt_returns_' + c for c in self.return_cols
                ]
                data[self.kurt_return_cols] = temp_kurt_returns
                data.reset_index(inplace=True)
                temp_kurt_returns = data[self.kurt_return_cols].values
                temp_kurt_returns = temp_kurt_returns.reshape(
                    (temp_kurt_returns.shape[0], temp_kurt_returns.shape[1],
                     1))
                data = data.drop(self.kurt_return_cols, axis=1)
                return temp_kurt_returns

            elif name == 'eqt_kurt_returns':
                temp_eqt_kurt_returns = data.groupby('eqt_code')[
                    self.return_cols].apply(pd.DataFrame.kurt)
                data.set_index(['eqt_code'], inplace=True)
                self.eqt_kurt_return_cols = [
                    'eqt_kurt_returns_' + c for c in self.return_cols
                ]
                data[self.eqt_kurt_return_cols] = temp_eqt_kurt_returns
                data.reset_index(inplace=True)
                temp_eqt_kurt_returns = data[self.eqt_kurt_return_cols].values
                temp_eqt_kurt_returns = temp_eqt_kurt_returns.reshape(
                    (temp_eqt_kurt_returns.shape[0],
                     temp_eqt_kurt_returns.shape[1], 1))
                data = data.drop(self.eqt_kurt_return_cols, axis=1)
                return np.nan_to_num(temp_eqt_kurt_returns)

            elif name == "return_diff_to_market_input":
                temp = self.scale(create_input('returns_input', data).squeeze() - create_input(
                    'market_returns_input', data).squeeze())
                return temp.reshape((temp.shape[0],temp.shape[1],1))

            elif name == "log_vol_input":
                temp_vol = np.log(np.abs(data[self.return_cols].values) + eps)
                temp_vol = self.scale(temp_vol)
                temp_vol = temp_vol.reshape((temp_vol.shape[0],
                                             temp_vol.shape[1], 1))
                return temp_vol

            elif name == "market_log_vol_input":
                temp_market_log_vol = (data.apply(lambda x: np.log(
                    np.abs(x) + eps)))[self.return_cols]
                temp_df = pd.DataFrame()
                temp_df['date'] = data['date']
                temp_df[self.return_cols] = temp_market_log_vol
                temp_market_log_vol = temp_df.groupby('date')[
                    self.return_cols].mean()
                temp_df.set_index(['date'], inplace=True)
                self.market_log_vol_cols = [
                    'market_log_vol_' + c for c in self.return_cols
                ]
                temp_df[self.market_log_vol_cols] = temp_market_log_vol
                temp_df.reset_index(inplace=True)
                temp_market_log_vol = temp_df[self.market_log_vol_cols].values
                temp_market_log_vol = self.scale(temp_market_log_vol)
                temp_market_log_vol = temp_market_log_vol.reshape(
                    (temp_market_log_vol.shape[0],
                     temp_market_log_vol.shape[1], 1))
                del temp_df
                return temp_market_log_vol

            elif name == "log_vol_diff_to_market_input":
                temp = create_input('log_vol_input',data) - create_input(
                    'market_log_vol_input',data)
                temp = self.scale(temp)
                return 
            elif name == 'eqt_avg_log_vol':
                temp_eqt_log_vol = data.groupby('eqt_code')[
                    self.return_cols].mean()
                data.set_index(['eqt_code'], inplace=True)
                self.eqt_log_vol_cols = [
                    'eqt_log_vol_' + c for c in self.return_cols
                ]
                data[self.eqt_log_vol_cols] = temp_eqt_log_vol
                data.reset_index(inplace=True)
                temp_eqt_log_vol = data[self.eqt_log_vol_cols].values
                temp_eqt_log_vol = temp_eqt_log_vol.reshape(
                    (temp_eqt_log_vol.shape[0], temp_eqt_log_vol.shape[1], 1))
                data = data.drop(self.eqt_log_vol_cols, axis=1)
                return temp_eqt_log_vol

            elif name == "handmade_features_input":
                temp_non_return_cols = [col for col in self.non_return_cols if 
                                        not col in ['return_nan',
                                                    'countd_date',
                                                    'countd_product']]
                
                return self.scale(data[temp_non_return_cols].values)
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
                    plateau_patience=10,
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
                "decay": 0,
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
            "factor": 0.9,
            "patience": plateau_patience,
            "min_lr": 10**-6
        }

        conf["metrics"] = ["acc"]

        early_stop, checkpointer, reduce_lr, clr = callbacks_intrain()
                
        self.model.compile(
            optimizer='adam',
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
