#!/usr/bin/env python
# coding: utf-8
import numpy as np

from sklearn.utils import shuffle

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam, SGD
from keras.utils import to_categorical
from keras.models import load_model, Sequential
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint


def sample_frames(dataset, interval=5):
    """
    Sample the video frame arrays by a given interval so that we can shrink the data scale 
    """
    dataset_tmp = []
    for video in dataset:
        dataset_tmp.append(video[::interval])

    return np.array(dataset_tmp)


if __name__ == '__main__':
    # load train set, validation set and test set from .npy files
    train_data = np.load('train_data.npy')
    train_label = np.load('train_label.npy')

    valid_data = np.load('valid_data.npy')
    valid_label = np.load('valid_label.npy')

    test_data = np.load('test_data.npy')
    test_label = np.load('test_label.npy')


    # sample datasets and convert lables to float type
    train_data = sample_frames(train_data)
    train_label = train_label.astype('float32')

    valid_data = sample_frames(valid_data)
    valid_label = valid_label.astype('float32')

    test_data = sample_frames(test_data)
    test_label = test_label.astype('float32')

    # shuffle train set
    train_data, train_label = shuffle(train_data, train_label)

    # define the input shape of the model using the train set shape
    frames, rows, columns, channels = train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4]

    # create VGG16+LSTM model
    model = Sequential()
    model.add(TimeDistributed(VGG16(input_shape=(rows, columns, channels), weights="imagenet", include_top=False)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))

    # set SGD optimizer and compile the model
    optimizer = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

    # set checkpoint for saving models automatically in every 2 epochs
    checkpoint = ModelCheckpoint('vgg16_lstm_model_{epoch:d}.h5', period=2) 

    # set validation set and train the model
    history = model.fit(train_data, train_label, epochs=30, batch_size=4, 
                        validation_data=(valid_data, valid_label), callbacks=[checkpoint])
