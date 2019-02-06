import utils
import features
from config import *
from sklearn.preprocessing import StandardScaler

class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

class Data:
    def __init__(self, split_val=0.1, scaler = None, seed=SEED, verbose=False):

        if verbose:
            print("Loading of the train dataset...")
        
        self.x, self.y, self.labels = utils.load_train()
        if verbose:
            print("Train dataset loaded!")
            print("Loading of the test dataset...")
            
        self.x_test = utils.load_test()
        if verbose:
            print("Test dataset loaded!")
            print("Add features...")
            
        features.add_features(self.x)
        features.add_features(self.x_test)
        
        if not scaler is None :
            
            if scaler == 'StandardScaler' :
                
                self.x.iloc[:,75::] = StandardScaler().fit_transform(self.x.iloc[:,75::])
                self.x_test.iloc[:,75::] = StandardScaler().fit_transform(self.x_test.iloc[:,75::])
            
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
    data = Data(verbose=True)
