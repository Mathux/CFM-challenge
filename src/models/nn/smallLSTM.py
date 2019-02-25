from keras.layers import (Dense, Dropout, Embedding, PReLU,
                          SpatialDropout1D, concatenate, Flatten, MaxPooling1D,
                          RepeatVector, LSTM, Bidirectional)
from keras.models import Model, Input

from src.models.nn.model import GeneralLSTM
from src.models.nn.janet import JANET
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



class smallLSTM(GeneralLSTM):
    def __init__(self,
                 data,
                 eqt_embeddings_size=10,
                 lstm_out_dim=64,
                 use_lstm=True,
                 dropout_rate=0.5,
                 dropout_spatial_rate=0.5,
                 dropout_lstm=0.5,
                 dropout_lstm_rec=0.5,
                 kernel_size=3,
                 loss='binary_crossentropy',
                 optimizer=None):
        super(smallLSTM, self).__init__(
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
        # First create the eqt embeddings
        eqt_code_input = Input(shape=[1], name='eqt_code_input')
        eqt_emb = Embedding(
            output_dim=self.eqt_embeddings_size,
            input_dim=self.n_eqt,
            input_length=1,
            name='eqt_embeddings')(eqt_code_input)
        eqt_emb = Dropout(self.dropout_spatial_rate)(eqt_emb)
        eqt_emb = Flatten()(eqt_emb)
        
        #eqt_emb_2 = RepeatVector(self.returns_length)(eqt_emb)
        
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
        
        #nb_days_eqt_traded = Input(shape=[1], name='nb_days_eqt_traded_input')
        context_eqt_day = concatenate([eqt_emb,nb_eqt_traded,nb_nans_data])
        context_eqt_day = Dense(64,activation = 'relu')(context_eqt_day)
        
        
        returns_input = Input(
            shape=(self.returns_length, 1), name='returns_input')
        market_returns_input = Input(
            shape=(self.returns_length, 1), name='market_returns_input')
        #kurt_returns_input = Input(
        #        shape=(self.returns_length, 1), name='kurt_returns_input')
        return_diff_to_market_input = Input(
                shape=(self.returns_length, 1), 
                name='return_diff_to_market_input')
        #market_kurt_returns_input = Input(
        #        shape=(self.returns_length, 1), name='market_kurt_returns_input')
        #eqt_avg_returns_input = Input(
        #        shape=(self.returns_length, 1), name='eqt_avg_returns_input')
        #eqt_kurt_returns_input = Input(
        #        shape=(self.returns_length, 1), name='eqt_kurt_returns_input')
        
        #market_returns_features = concatenate([market_returns_input,market_kurt_returns_input,return_diff_to_market_input])
        #eqt_returns_features = concatenate([returns_input,kurt_returns_input,eqt_avg_returns_input,eqt_kurt_returns_input]) 
        
        #temp = concatenate([eqt_emb_2,returns_input])
        returns_features_lstm = JANET(
            self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = True)(returns_input)
        
        x = concatenate([returns_input,market_returns_input,return_diff_to_market_input])
        
        temporal_features = JANET(self.lstm_out_dim,
            return_sequences=False,
            dropout=self.dropout_lstm,
            recurrent_dropout=self.dropout_lstm_rec, unroll = True)(x)
        
        
        temporal_features = concatenate([returns_features_lstm,temporal_features])
        
        temporal_features = Dense(64,activation = 'relu')(temporal_features)
        
        x = concatenate([returns_features_lstm,context_eqt_day,temporal_features])
        
        x = Dense(64,activation = 'relu')(x)
        
        x = Dropout(self.dropout_rate)(x)
        
        output = Dense(2,activation = 'softmax',name = 'output')(x)
        
        
        
        #x_returns = concatenate([market_returns_features,eqt_returns_features])
        
        #temp_returns_features = JANET(self.lstm_out_dim//2,
        #    return_sequences=False,
        #    dropout=self.dropout_lstm,
        #    recurrent_dropout=self.dropout_lstm_rec)(x_returns)
        
        #temp_returns_features = PReLU()(temp_returns_features)
        
        
        
        
        #log_vol_input = Input(shape=(self.returns_length, 1), name='log_vol_input')
        #market_log_vol_input = Input(shape=(self.returns_length, 1), name='market_log_vol_input')
        #log_vol_diff_to_market_input = Input(shape=(self.returns_length, 1), name='log_vol_diff_to_market_input')
        #kurt_log_vol_input = Input(
        #        shape=(self.returns_length, 1), name='kurt_returns_input')
        #log_vol_diff_to_market_input = Input(
        #       shape=(self.returns_length, 1), 
        #       name='return_diff_to_market_input')
        #market_kurt_log_vol_input = Input(
        #        shape=(self.returns_length, 1), name='market_kurt_returns_input')
        #eqt_avg_log_vol_input = Input(
        #        shape=(self.returns_length, 1), name='eqt_avg_returns_input')
        #eqt_kurt_log_vol_input = Input(
        #        shape=(self.returns_length, 1), name='eqt_kurt_returns_input')
        
        #market_log_vol_features = concatenate([market_log_vol_input,market_kurt_log_vol_input,log_vol_diff_to_market_input])
        #eqt_log_vol_features = concatenate([log_vol_input,kurt_log_vol_input,eqt_avg_log_vol_input,eqt_kurt_log_vol_input]) 

        #temp = concatenate([eqt_emb_2,vol_input])
        #log_vol_features_lstm = Bidirectional(LSTM(
        #    self.lstm_out_dim,
        #    return_sequences=True,
        #    dropout=self.dropout_lstm,
        #    recurrent_dropout=self.dropout_lstm_rec))(log_vol_input)
        
        
        #x = concatenate([returns_features_lstm, log_vol_features_lstm])

        #temporal_features = LSTM(self.lstm_out_dim,return_sequences=False,dropout=self.dropout_lstm,recurrent_dropout=self.dropout_lstm_rec)(x)
        #pred_temporal = Dense(2, activation='softmax', name='pred_temporal')(temporal_features)
        
        #x = concatenate([context_eqt_day,temporal_features,returns_features])
        #x = Dense(128,activation = 'relu')(x)
        #x = Dropout(self.dropout_rate)(x)
        #pred_temporal = Dense(2, activation='sigmoid',name = 'pred_temporal')(x)

        model = Model(
            inputs=[eqt_code_input, 
                    nb_eqt_traded_input, 
                    nb_nan_input, 
                    returns_input, 
                    market_returns_input, 
                    return_diff_to_market_input],
            outputs=[output])

        inputs = ["eqt_code_input", 
                  "nb_eqt_traded", 
                  "nb_nans_data", 
                  "returns_input", 
                  "market_returns_input",
                  "return_diff_to_market_input"]
        return model, inputs
    
    def compile_fit(self,
                    checkpointname,
                    epochs=30,
                    plateau_patience=1,
                    batch_size=4096,
                    verbose=0):

        conf = {}

        if self.optimizer is None:
            conf["optimizer"] = {
                "lr": 0.01,
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

        conf["EarlyStopping"] = {
            "monitor": "val_loss",
            "patience": 30
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
            "factor": 0.9,
            "patience": plateau_patience,
            "min_lr": 10**-20
        }

        reduce_lr = ReduceLROnPlateau(
            monitor=conf["ReduceLROnPlateau"]["monitor"],
            factor=conf["ReduceLROnPlateau"]["factor"],
            patience=conf["ReduceLROnPlateau"]["patience"],
            min_lr=conf["ReduceLROnPlateau"]["min_lr"],
            verbose=verbose)

        conf["metrics"] = ["acc"]
        
        self.model.compile(
            optimizer=opti, 
            loss= {'output' : self.loss},
            metrics= {'output' : conf["metrics"]},
            loss_weights=[1])

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


if __name__ == '__main__':
    from src.tools.experiment import Experiment
    from src.tools.dataloader import Data
    from src.tools.utils import plot_training
    from keras.utils import plot_model
    
    exp = Experiment(modelname="small_janet")
    data = Data(small=False, verbose=True, ewma = False, aggregate = True)
    exp.addconfig("data", data.config)

    model = smallLSTM(data, use_lstm=True)
    exp.addconfig("model", model.config)

    plot_model(model.model, to_file=exp.pnggraph, show_shapes=True)

    model.model.summary()
    # Fit the model
    history = model.compile_fit(
        checkpointname=exp.modelname, epochs=30, plateau_patience=1, verbose=1)

    exp.addconfig("learning", model.learning_config)
    exp.saveconfig(verbose=True)

    plot_training(
        history, show=False, losspath=exp.pngloss, accpath=exp.pngacc)

    #model.predict_test(exp.allpath("predictions.csv"))

#%%