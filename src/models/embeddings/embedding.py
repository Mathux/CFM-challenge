import pickle

import numpy as np
from keras import backend as K
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Reshape
from keras.models import Sequential
from sklearn.utils import resample

import src.config as config
from src.tools.dataloader import Data, Dataset


# Class to do embedding on eqt
class EqtEmbedding():
    def __init__(self, data, outputdim, maxvalue=1511, verbose=False):
        # Parameters
        self.outputdim = outputdim

        if verbose:
            print("Embedding: loading train vectors and labels...")

        self.train = Dataset(
            *self.load_data(data.train.data, data.train.labels, maxvalue))
        if verbose:
            print("Train vector and labels loaded!")
            print("Embedding: loading val vectors and labels...")

        self.val = Dataset(
            *self.load_data(data.val.data, data.val.labels, maxvalue))
        if verbose:
            print("Val vector and labels loaded!")

        self.n_eqt, self.n_date, self.n_hours = self.train.data.shape
        self.eqt_code = data.train.data['eqt_code'].unique()
        # self.n_eqt, self.n_date, self.n_hours = (680, 1511, 71)

    # Return the embedding
    def transform(self, opti, loss, batch_size=32, epochs=20):
        self.embeddings = {}
        for i in range(self.n_eqt):
            print('Fitting equity : ', self.eqt_code[i], '({}/{})'.format(
                i, self.n_eqt))
            model = self.create_model(opti, loss)
            try:
                self.history = model.fit(
                    self.train.data[i],
                    self.train.labels[i],
                    batch_size=batch_size,
                    validation_data=(self.val.data[i], self.val.labels[i]),
                    epochs=epochs,
                    verbose=1)
            except IndexError:
                self.history = model.fit(
                    self.train.data[i],
                    self.train.labels[i],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

            layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
            self.embeddings[self.eqt_code[i]] = np.mean(
                layer_output([self.train.data[i]])[0], axis=0)

    # Save the weights
    def save_weight(self, name):
        self.model.save(name)

    # Dico eqt_code/300 values
    def save_embeddings(self, name):
        with open(name, 'wb') as handle:
            pickle.dump(
                self.embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_embeddings(self, name):
        with open(name, 'rb') as handle:
            self.embeddings = pickle.load(handle)

    # Process the data
    def load_data(self, data, labels, maxvalue=1511):

        data = data.merge(labels, on='ID')
        return_cols = [col for col in data.columns if col.endswith(':00')
                       ] + ["ID", 'end_of_day_return']

        eqt_codes = data["eqt_code"].unique()
        n = len(eqt_codes)

        processed_vector = np.zeros((n, maxvalue, len(return_cols) - 2),
                                    dtype='float64')
        processed_labels = np.zeros((n, maxvalue), dtype=np.int64)

        # For each eqt, process it
        for i in range(n):
            # get all the returns
            vector_eqt = data[data["eqt_code"] ==
                              eqt_codes[i]][return_cols].values

            # Fill the vector to have a fixed size by resampling
            ndate = vector_eqt.shape[0]
            more_vector = resample(
                vector_eqt,
                n_samples=maxvalue - ndate,
                random_state=config.SEED)
            vector = np.concatenate((vector_eqt, more_vector))

            # Shuffle it
            np.random.shuffle(vector)

            final_labels = vector[:, -1]
            final_vector = vector[:, :-2]  # discard the id
            #id_eqt = np.array(vector[:,-2], dtype=np.int64) # keep the id

            ## too long in time
            #final_labels = np.array([labels[labels["ID"] == x].values[0][1] for x in id_eqt])

            # better solution but not perfect yet
            #tmp_labels = labels[labels["ID"].isin(id_eqt)]
            #final_labels = np.array([tmp_labels[tmp_labels["ID"] == x].values[0][1] for x in id_eqt])

            ## Fill the large vector
            processed_vector[i] = final_vector

            #final_labels = to_categorical(final_labels)
            processed_labels[i] = final_labels

        return processed_vector, processed_labels


# Some tests with a very basic embedding, it didn't care about the temporality
class NaiveEmbedding(EqtEmbedding):
    def __init__(self,
                 data,
                 outputdim=300,
                 opti="adam",
                 lr=0.1,
                 loss="binary_crossentropy",
                 verbose=False):
        super(NaiveEmbedding, self).__init__(data, outputdim, verbose=verbose)
        self.model = self.create_model(opti, loss)

    def create_model(self, opti, loss):

        model = Sequential()
        model.add(Dense(self.outputdim, input_dim=self.n_hours))
        # Now we have the embedding, continue to learn
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=opti, loss=loss, metrics=['acc'])

        return model
    
class LessNaiveEmbedding(EqtEmbedding):
    def __init__(self,
                 data,
                 outputdim=300,
                 opti="adam",
                 lr=0.1,
                 loss="binary_crossentropy",
                 verbose=False):
        super(LessNaiveEmbedding, self).__init__(data, outputdim, verbose=verbose)
        self.model = self.create_model(opti, loss)

    def create_model(self, opti, loss):

        model = Sequential()
        model.add(Reshape((self.n_hours,1)))
        model.add(TimeDistributed(Dense(self.outputdim)))
        
        model.add(LSTM(32))
        # Now we have the embedding, continue to learn
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=opti, loss=loss, metrics=['acc'])

        return model


if __name__ == '__main__':
    pass
#    data = Data(small=True, verbose=True)
#    embeddings_model = LessNaiveEmbedding(data, verbose=True)
#    embeddings_model.transform('adam', 'binary_crossentropy')
#    embeddings_model.save_embeddings('embeddings.pickle')

#%%