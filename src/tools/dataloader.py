import sklearn.preprocessing

import src.tools.features as features
import src.tools.utils as utils
import src.config as config


class Dataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


class Data:
    def __init__(self,
                 d=233,
                 split=True,
                 kfold=-1,
                 split_val=0.1,
                 scaler='StandardScaler',
                 seed=config.SEED,
                 verbose=False,
                 small=False,
                 embeddings=None,
                 ewma=False,
                 aggregate=False):
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

        # print("mem self.x:", id(self.x))
        self.x = features.add_features(
            self.x, embeddings=embeddings, ewma=ewma, aggregate=aggregate)
        self.x_test = features.add_features(
            self.x_test, embeddings=embeddings, ewma=ewma, aggregate=aggregate)

        if verbose:
            print("Features added!")

        if scaler is not None:
            if scaler == 'StandardScaler':
                scaled_columns = [
                    c for c in self.x.columns
                    if ((not c.endswith(':00')) and (c not in [
                        'eqt_code', 'ID', 'sector', 'date', 'return_nan',
                        'countd_date', 'countd_product', 'Unnamed: 0'
                    ]))
                ]
                self.x[scaled_columns] = sklearn.preprocessing.StandardScaler(
                ).fit_transform(self.x[scaled_columns])
                self.x_test[
                    scaled_columns] = sklearn.preprocessing.StandardScaler(
                    ).fit_transform(self.x_test[scaled_columns])

        self.nunique = self.x['eqt_code'].nunique()
        self.eqt_list = list(self.x['eqt_code'].unique())

        if split:
            if verbose:
                print("Split the dataset...")

            if kfold > 0:
                if verbose:
                    print("Using kfold with k=", kfold)
                self.split(split_val, seed, kfold=kfold)
            else:
                self.split(split_val, seed)

            if verbose:
                print("Dataset splitted!")

        self.test = Dataset(self.x_test, None)

        if verbose:
            print("Data loading done!")

        self.config = {
            "split": split,
            "split_val": split_val,
            "kfold": kfold,
            "scaler": scaler,
            "seed": seed,
            "small": small,
            "embeddings": embeddings,
            "ewma": ewma
        }

    def split(self, split_val, seed, kfold=None):
        self.kfold = kfold

        # keep that for compatibility
        train, val, train_labels, val_labels = utils.split_dataset(
            self.x, self.labels, split_val, seed)
        self.train = Dataset(train, train_labels)
        self.val = Dataset(val, val_labels)

        if kfold is not None:
            folds, folds_label = utils.kfold_split_dataset(
                self.x, self.labels, kfold, seed)
            self.folds = []
            for f, fl in zip(folds, folds_label):
                self.folds.append(Dataset(f, fl))

    def merge_folds(self, k):
        import pandas as pd
        data = pd.concat(
            tuple(
                [self.folds[i].data for i in range(self.kfold) if not i == k]))
        labels = pd.concat(
            tuple(
                [self.folds[i].labels for i in range(self.kfold) if not i == k]))
        return data, labels


if __name__ == '__main__':
    data = Data(small=True, verbose=True, kfold=4)
