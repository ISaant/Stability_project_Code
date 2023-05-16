#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:06:41 2023

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os 
import tensorflow as tf
# from Test_loadModel import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
# Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
# Memory growth must be set before GPUs have been initialized
        print(e)
        
        
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model

from sklearn import metrics as skmetrics
# from TrueFalse import TrueFalse as TF



# def ParallelNN(Split,Dir,winsize,Ndatos,Name):
# Data=PCA
# X_train, X_test, y_train, y_test = train_test_split(Data, Ca, test_size=0.33, random_state=42)

    