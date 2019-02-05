import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from sklearn.utils import resample

from processing_data import Data, Dataset
from config import *


# Class to do embedding on eqt
class EqtEmbedding():
    def __init__(self, data, outputdim, maxvalue=1511, verbose=False):
        # Parameters
        self.outputdim = outputdim

        if verbose:
            print("Embedding: loading train vectors and labels...")
            
        self.train = Dataset(*self.load_data(data.train.data, data.train.labels, maxvalue))
        if verbose:
            print("Train vector and labels loaded!")
            print("Embedding: loading val vectors and labels...")
            
        self.val = Dataset(*self.load_data(data.val.data, data.val.labels, maxvalue))
        if verbose:
            print("Val vector and labels loaded!")
            
        self.n_eqt, self.n_date, self.n_hours = self.train.data.shape
        #self.n_eqt, self.n_date, self.n_hours = (680, 1511, 71)

    def fit(self, batch_size=32, epochs=2):
        self.history = self.model.fit(self.train.data, self.train.labels, batch_size=batch_size,
                                      validation_data=(self.val.data, self.val.labels), epochs=epochs)
        
    # Return the embedding
    def transform(self, x):
        self.model.predict()
        pass

    # Save the weights
    def save_weight(self, name):
        pass #self.model.save(name)

    # Dico eqt_code/300 values
    def save_embeddings(self, name):
        pass


    # Process the data
    def load_data(self, data, labels, maxvalue=1511):        
        return_cols = [col for col in data.columns if col.endswith(':00')] + ["ID"]
        
        eqt_codes = data["eqt_code"].unique()
        n = len(eqt_codes)

        processed_vector = np.zeros((n, maxvalue, len(return_cols)-1), dtype='float64')
        processed_labels = np.zeros((n, maxvalue), dtype=np.int64)
        
        # Foreach eqt, process it
        for i in range(n):
            # get all the returns
            vector_eqt = data[data["eqt_code"] == eqt_codes[i]][return_cols].values

            # Fill the vector to have a fixed size by resampling
            ndate = vector_eqt.shape[0]
            more_vector = resample(vector_eqt, n_samples=maxvalue-ndate, random_state=SEED)
            vector = np.concatenate((vector_eqt, more_vector))

            # Shuffle it
            np.random.shuffle(vector)

            final_vector = vector[:,:-1] # discard the id
            id_eqt = np.array(vector[:,-1], dtype=np.int64) # keep the id
            
            ## too long in time
            #final_labels = np.array([labels[labels["ID"] == x].values[0][1] for x in id_eqt])

            # better solution but not perfect yet
            tmp_labels = labels[labels["ID"].isin(id_eqt)]
            final_labels = np.array([tmp_labels[tmp_labels["ID"] == x].values[0][1] for x in id_eqt])

            ## Fill the large vector
            processed_vector[i] = final_vector
            processed_labels[i] = final_labels
        return processed_vector, processed_labels
        
    
    def print_model(self):
        print(self.model.summary())


# Some tests with a very basic embedding, it didn't care about the temporality
class NaiveEmbedding(EqtEmbedding):
    def __init__(self, data, outputdim=300, opti="adam", lr=0.1, loss="mse", verbose=False):
        super(NaiveEmbedding, self).__init__(data, outputdim, verbose=verbose)
        self.model = self.create_model(opti, loss)
        
    def create_model(self, opti, loss):
        model = Sequential()
        model.add(Flatten(input_shape=(self.n_date, self.n_hours)))
        model.add(Dense(self.n_date, activation='relu'))
        model.add(Dense(self.outputdim, activation='relu'))
        
        # Now we have the embedding, continue to learn
        model.add(Dense(self.outputdim, activation='relu'))
        model.add(Dense(n_date, activation='softmax'))
        
        model.compile(optimizer=opti, loss=loss, metrics=['acc'])
        return model



if __name__ == '__main__':
    data = Data(verbose=True)
    embedding = NaiveEmbedding(data, verbose=True)
