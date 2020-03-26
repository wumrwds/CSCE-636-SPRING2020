#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np
import math


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.utils import to_categorical
from keras.models import load_model


def video_to_frames(video):
    # extract frames from a video and return a frame array
    vidcap = cv2.VideoCapture(video)
    frames = []
    while vidcap.isOpened():
        success, image = vidcap.read()
        
        if success:
            frames.append(image)
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
    
    return np.array(frames)



# get the minimum frames length
frame_lens = []
for folder in os.listdir('dataset/train'):
    folder_path = 'dataset/train/' + folder
    for filename in os.listdir(folder_path):
        frames = video_to_frames(folder_path + '/' + filename)
        frame_lens.append(len(frames))

np.array(frame_lens).min()


# get train set
train_data = []
train_label = []
label = 0
for folder in os.listdir('dataset/train'):
    folder_path = 'dataset/train/' + folder
    for filename in os.listdir(folder_path):
        frames = video_to_frames(folder_path + '/' + filename)
        frames = frames[math.floor(len(frames)/2) - 44 : math.floor(len(frames)/2) + 44]
        print(filename, label)
        train_data.append(frames)
        train_label.append(label)

    label += 1

train_data = np.array(train_data)
train_label = np.array(train_label)
train_label = to_categorical(train_label, num_classes=label)


# get test set
test_data = []
test_label = []
label = 0
for folder in os.listdir('dataset/test'):
    folder_path = 'dataset/test/' + folder
    for filename in os.listdir(folder_path):
        frames = video_to_frames(folder_path + '/' + filename)
        frames = frames[math.floor(len(frames)/2) - 44 : math.floor(len(frames)/2) + 44]
        
        new_frames = np.zeros([len(frames), 240,320,3])
        for i in range(len(frames)):
            new_frames[i] = cv2.resize(frames[i], dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
        
        print(filename, label)
        test_data.append(new_frames)
        test_label.append(label)

    label += 1

test_data = np.array(test_data)
test_label = np.array(test_label)
test_label = to_categorical(test_label, num_classes=label)


# create model
classes = label
frames, rows, columns, channels = train_data.shape[1], train_data.shape[2], train_data.shape[3], train_data.shape[4]

video = Input(shape=(frames, rows, columns, channels))
cnn_base = VGG16(input_shape=(rows, columns, channels), weights="imagenet", include_top=False)
cnn_out = GlobalAveragePooling2D()(cnn_base.output)
cnn = Model(input=cnn_base.input, output=cnn_out)
cnn.trainable = False
encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(256)(encoded_frames)
hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
outputs = Dense(output_dim=classes, activation="softmax")(hidden_layer)
model = Model([video], outputs)
optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 


# train
history = model.fit(train_data, train_label, epochs=3, batch_size=30)

# save model
model.save(filepath='my_model.h5')

# predict
prediction = model.predict(test_data)

# show predict result
for p in prediction:
    print(p.argmax())

