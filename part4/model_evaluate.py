#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np
import math
import sys

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.utils import to_categorical
from keras.models import load_model


print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

standard_labels = ["faint", "wonder", "car", "walk", "crouch", "bend", "jump", "run"]
test_label = -1
for i in range(len(standard_labels)):
    if standard_labels[i] in sys.argv[0]:
        test_label = i
        break


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


frames = video_to_frames(sys.argv[0])
if len(frames) > 88:
    frames = frames[math.floor(len(frames)/2) - 44 : math.floor(len(frames)/2) + 44]

new_frames = np.zeros([max(len(frames), 88), 240,320,3])
for i in range(len(frames)):
    new_frames[i] = cv2.resize(frames[i], dsize=(320, 240), interpolation=cv2.INTER_CUBIC)

print(sys.argv[0], test_label)

test_data = np.array([new_frames])
print(test_data.shape)


# load model
model = load_model('my_model.h5')

# predict
prediction = model.predict(test_data)

# show predict result
for p in prediction:
    print("The prediction result is: %s" % standard_labels[p.argmax()])

