#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:15:21 2019

@author: evrardgarcelon, mathispetrovich
"""

import sys
import os

import numpy as np
import pandas as pd
import pylab as plt

import src.config as conf

from sklearn.manifold import TSNE


# Create a small dataset
def create_small_dataset():
    paths = [conf.path_to_test, conf.path_to_train, conf.path_to_train_returns]
    paths_small = [
        conf.path_to_test_small, conf.path_to_train_small,
        conf.path_to_train_returns_small
    ]

    ntrain = 5000
    ntest = 1000
    numbers = [ntest, ntrain, ntrain]

    for path, path_small, number in zip(paths, paths_small, numbers):
        data = pd.read_csv(path)
        data = data.head(number)
        with open(path_small, "w") as f:
            f.write(data.to_csv())


# Load train data
def load_train(small=False):
    trainPath = conf.path_to_train_small if small else conf.path_to_train
    trainPathReturns = conf.path_to_train_returns_small if small else conf.path_to_train_returns

    x_train = pd.read_csv(trainPath)
    y_train = pd.read_csv(trainPathReturns, sep=',')

    y_train_labels = y_train.copy()
    y_train_labels['end_of_day_return'] = 1 * (y_train['end_of_day_return'] >=
                                               0)

    return x_train, y_train, y_train_labels


# Load test data
def load_test(small=False):
    testPath = conf.path_to_test_small if small else conf.path_to_test
    x_test = pd.read_csv(testPath)
    return x_test


# Split the train dataset into training and validation (keep different date)
def split_dataset(data, labels, split_val=0.1, seed=conf.SEED):
    np.random.seed(seed)

    data = data.merge(labels, on='ID')

    dates = data["date"].unique().copy()
    n_dates = len(dates)
    all_index = np.arange(n_dates)
    np.random.shuffle(all_index)

    train_index = all_index[int(split_val * n_dates):]
    val_index = all_index[0:int(split_val * n_dates)]

    train = data[data["date"].isin(dates[train_index])]
    val = data[data["date"].isin(dates[val_index])]

    train_labels = train[['ID', 'end_of_day_return']]
    val_labels = val[['ID', 'end_of_day_return']]

    train = train.drop('end_of_day_return', axis=1)
    val = val.drop('end_of_day_return', axis=1)

    return train, val, train_labels, val_labels


# Split the train dataset into kfolds
def kfold_split_dataset(data, labels, k, seed=conf.SEED):
    np.random.seed(seed)

    data = data.merge(labels, on='ID')

    dates = data["date"].unique().copy()
    n_dates = len(dates)
    all_index = np.arange(n_dates)
    np.random.shuffle(all_index)

    nb_el_per_fold = int(n_dates//k)
    folds = []
    folds_label = []
    for i in range(k):
        fold_index = all_index[nb_el_per_fold*i:nb_el_per_fold*(i+1)]
        fold = data[data["date"].isin(dates[fold_index])]
        fold_labels = fold[['ID', 'end_of_day_return']]
        fold = fold.drop('end_of_day_return', axis=1)

        folds.append(fold)
        folds_label.append(fold_labels)

    return folds, folds_label


# Tools to give the csv format
def submission(prediction, ID=None, name="predictions.csv"):
    if isinstance(prediction, pd.core.frame.DataFrame):
        prediction.to_csv(name, index=False)
    else:
        pred = pd.DataFrame()
        pred['ID'] = ID
        pred['end_of_day_return'] = prediction
        pred.to_csv(name, index=False)


# Give some cool bar when we are waiting
def progressBar(value, endvalue, bar_length=50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n Progress: [{0}] {1}%".format(
        arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def plot_tsne(data):
    from sklearn.metrics.pairwise import cosine_similarity
    tsne = TSNE(n_components=2, learning_rate= 1000, metric = cosine_similarity)
    X_r = tsne.fit_transform(data)
    plt.scatter(X_r[:, 0], X_r[:, 1])
    plt.show()


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_training(history, show=True, losspath=None, accpath=None):
    plt.figure()
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    if losspath:
        plt.savefig(losspath)
    if show:
        plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    if accpath:
        plt.savefig(accpath)

    if show:
        plt.show()


if __name__ == "__main__":
    pass
    # create_small_dataset()
