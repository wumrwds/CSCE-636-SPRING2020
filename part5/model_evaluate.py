#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
import math
import sys

from data_preprocessing import video_to_frames, normalize_frame_array
from model_train import sample_frames

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

# get input arguments
model_name = sys.argv[0]
video_path = sys.argv[1]

# load model
model = load_model(model_name)
frame_length = model.layers[0].input_shape[0] * 5

# parse video to frame array
frames = video_to_frames(video_path)

# normalize frame array
test_data = normalize_frame_array(frames, frame_length)
test_data = sample_frames(test_data)

# predict
print("Start predicting...")
prediction = model.predict(test_data)
print("The model prediction result (0 represents non-slip, 1 represents slip) is: %d" % (1 if prediction[0] > 0.5 else 0))
