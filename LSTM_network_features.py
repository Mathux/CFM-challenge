#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:57:03 2019

@author: evrardgarcelon
"""

import numpy as np
import pylab as plt

from processing_data import *
from keras.models import Model, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam
from keras.layers.advanced_activations import *
from keras.layers.recurrent import LSTM
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import LeakyReLU

from janet import JANET

data = Data(small = True,scaler = None)

return_cols = [c for c in data.train.data.columns if c.endswith(':00')]
X_train = data.train.data[return_cols].values
y_train = to_categorical(data.train.labels['end_of_day_return'].values)
X_val = data.val.data[return_cols].values
y_val= to_categorical(data.val.labels['end_of_day_return'].values)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_val = X_val.reshape((X_val.shape[0],X_val.shape[1],1))

model_inp = Input(shape=(71,1))
model_out = LSTM(50,return_sequences=False)(model_inp)
model_out = Dense(64)(model_out) 
model_out = BatchNormalization()(model_out)
model_out = Dropout(0.5)(model_out) 
model_out = LeakyReLU()(model_out)
model_out = Dense(2)(model_out)
model_out = Activation('softmax')(model_out)

model = Model(inputs = model_inp, outputs = model_out)
opt = Nadam(lr=0.001)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
          epochs = 20, 
          batch_size = 128, 
          verbose=1, 
          validation_data=(X_val, y_val),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()