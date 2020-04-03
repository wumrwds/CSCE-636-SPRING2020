#!/usr/bin/env python
# coding: utf-8
import numpy as np

from keras.models import load_model

for epoch in range(2, 30, 2):
    model = load_model("vgg16_lstm_model_%d.h5" % epoch)
    score, acc = model.evaluate(test_data, test_label, batch_size=4)
    print("Epoch #%d: score = %f, accuracy = %f" % (epoch, score, acc))