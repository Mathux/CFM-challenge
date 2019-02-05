import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from sklearn.utils import resample


from config import *
import utils


# Class to do embedding on eqt
class EqtEmbedding(ABC):
    def __init__(self, data, outputdim):
        # Parameters
        self.inputdim = inputdim

        self.data = self..load_data(data, 1511)
        self.n_eqt, self.n_date, self.n_hours = self.data.shape

    def train(self, data, labels):
        self.model.fit(data, labels)
        
    # Return the embedding
    def transform(self, x):
        self.model.predict()
        pass

    # Process the data
    def load_data(data, maxvalue=1511):
        return_cols = [col for col in data.columns if col.endswith(':00')]

        eqt_codes = data["eqt_code"].unique()
        n = len(eqt_codes)

        processed_vector = np.zeros((n, maxvalue, len(return_cols)), dtype='float64')
        
        # Foreach eqt, process it
        for i in range(n):
            # get all the returns
            vector_eqt = data[data["eqt_code"] == eqt_codes[i]][return_cols].values

            # Fill the vector to have a fixed size by resampling
            ndate, _ = vector_eqt.shape
            more_vector = resample(vector_eqt, n_samples=maxvalue-ndate, random_state=SEED)
            vector = np.concatenate((vector_eqt, more_vector))

            # Shuffle it
            np.random.shuffle(vector)

            ## Fill the large vector
            processed_vector[i] = vector
        return processed_vector
        
        
    def print_model(self):
        print(self.model.summary())


# Some tests with a very basic embedding, it didn't care about the temporality
class NaiveEmbedding(EqtEmbedding):
    def __init__(self, data, outputdim=300, opti="adam", lr=0.1, loss="binary_crossentropy"):
        super(NaiveEmbedding, self).__init__(data, outputdim)        
        self.model = self.create_model(opti, loss)
        
    def create_model(self, opti, loss):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(self.n_date, input_shape=(self.n_date * self.n_hours), activation='sigmoid'))
        model.add(Dense(self.outputdim, activation='sigmoid'))
        model.compile(optimizer=opti, loss=loss, metrics=['acc'])
        return model



if __name__ == '__main__':
    utils.load_data()
    embedding = NaiveEmbedding(20)
