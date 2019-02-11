import utils
import features
from config import *

from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

class Data:
    def __init__(self, split_val=0.1, scaler = 'StandardScaler', seed=SEED, verbose=False, small=False, embeddings = None):
        if small:
            print("Warning! Using small datasets..")
            
        if verbose:
            print("Loading of the train dataset...")
        
        self.x, self.y, self.labels = utils.load_train(small)
        if verbose:
            print("Train dataset loaded!")
            print("Loading of the test dataset...")
            
        self.x_test = utils.load_test(small)
        if verbose:
            print("Test dataset loaded!")
            print("Add features...")
            
        features.add_features(self.x, embeddings = embeddings)
        features.add_features(self.x_test, embeddings = embeddings)
        
        if not scaler is None :            
            if scaler == 'StandardScaler' :
                scaled_columns = [c for c in self.x.columns if ((not c.endswith(':00')) and (not c in ['eqt_code','ID','sector','date','return_nan','countd_date','countd_product']))]
                print(scaled_columns)
                self.x[scaled_columns] = StandardScaler().fit_transform(self.x[scaled_columns])
                self.x_test[scaled_columns] = StandardScaler().fit_transform(self.x_test[scaled_columns])
            
        if verbose:
            print("Features added!")
            print("Split the dataset...")
            
        self.split(split_val, seed)
        if verbose:
            print("Dataset splitted!")
            
        self.test = Dataset(self.x_test, None)

        if verbose:
            print("Data loading done!")
        
    def split(self, split_val, seed):
        train, val, train_labels, val_labels = utils.split_dataset(self.x, self.labels, split_val, seed)
        self.train = Dataset(train, train_labels)
        self.val = Dataset(val, val_labels)


if __name__ == '__main__':
    data = Data(small=True, verbose=True)
