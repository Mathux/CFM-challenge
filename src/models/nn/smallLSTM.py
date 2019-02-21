from keras.layers import (Dense, Dropout, Embedding, Conv1D, PReLU,
                          SpatialDropout1D, concatenate, Flatten, MaxPooling1D)
from keras.models import Model, Input

from src.models.nn.model import GeneralLSTM
from src.models.nn.janet import JANET


class smallLSTM(GeneralLSTM):
    def __init__(self,
                 data,
                 eqt_embeddings_size=8,
                 lstm_out_dim=32,
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
            input_dim=self.n_eqt + 1,
            input_length=1,
            name='eqt_embeddings')(eqt_code_input)
        eqt_emb = SpatialDropout1D(self.dropout_spatial_rate)(eqt_emb)
        eqt_emb = Flatten()(eqt_emb)
        # eqt_emb_2 = RepeatVector(self.returns_length)(eqt_emb)

        # Then the LSTM/CNN1D for the returns time series
        returns_input = Input(
            shape=(self.returns_length, 1), name='returns_input')
        if self.use_lstm:
            # temp = concatenate([eqt_emb_2,returns_input])
            returns_lstm = JANET(
                self.lstm_out_dim,
                return_sequences=False,
                dropout=self.dropout_lstm,
                recurrent_dropout=self.dropout_lstm_rec)(returns_input)

        else:
            returns_lstm = Conv1D(
                filters=self.lstm_out_dim,
                kernel_size=self.kernel_size,
                activation='linear',
                name='returns_conv')(returns_input)
            returns_lstm = Dropout(self.dropout_rate)(returns_lstm)
            returns_lstm = MaxPooling1D()(returns_lstm)
            returns_lstm = Flatten()(returns_lstm)

        # and the the LSTM/CNN part for the volatility time series
        vol_input = Input(shape=(self.returns_length, 1), name='vol_input')

        if self.use_lstm:
            # temp = concatenate([eqt_emb_2,vol_input])
            vol_lstm = JANET(
                self.lstm_out_dim,
                return_sequences=False,
                dropout=self.dropout_lstm,
                recurrent_dropout=self.dropout_lstm_rec)(vol_input)

        else:
            vol_lstm = Conv1D(
                filters=self.lstm_out_dim,
                kernel_size=self.kernel_size,
                activation='linear',
                name='vol_conv')(vol_input)
            vol_lstm = Dropout(self.dropout_rate)(vol_lstm)
            vol_lstm = MaxPooling1D()(vol_lstm)
            vol_lstm = Flatten()(vol_lstm)

        x = concatenate([eqt_emb, returns_lstm, vol_lstm])
        x = Dense(32, activation='linear')(x)
        x = Dropout(self.dropout_rate)(x)
        x = PReLU()(x)
        output = Dense(2, activation='softmax')(x)

        model = Model(
            inputs=[eqt_code_input, returns_input, vol_input],
            outputs=[output])

        inputs = ["eqt_code_input", "returns_input", "vol_input"]
        return model, inputs


if __name__ == '__main__':
    from src.tools.experiment import Experiment
    from src.tools.dataloader import Data
    from src.tools.utils import plot_training
    from keras.utils import plot_model
    
    exp = Experiment(modelname="small_janet")
    data = Data(small=True, verbose=True)
    exp.addconfig("data", data.config)

    model = smallLSTM(data, use_lstm=True)
    exp.addconfig("model", model.config)

    plot_model(model.model, to_file=exp.pnggraph, show_shapes=True)

    model.model.summary()
    # Fit the model
    history = model.compile_fit(
        checkpointname=exp.modelname, epochs=4, plateau_patience=10, verbose=1)

    exp.addconfig("learning", model.learning_config)
    exp.saveconfig(verbose=True)

    plot_training(
        history, show=False, losspath=exp.pngloss, accpath=exp.pngacc)

    model.predict_test(exp.allpath("predictions.csv"))
