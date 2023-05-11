#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:26:03 2023

@author: isaac
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

physical_devices= tf.config.experimental.list_physical_devices('GPU')
print ("Num GPUs Available: ",len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0],True)