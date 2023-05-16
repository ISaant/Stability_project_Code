#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:26:03 2023

@author: isaac
"""

import tensorflow as tf
import scipy 
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
# # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
# # Memory growth must be set before GPUs have been initialized
#         print(e)

from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler

#%% Perceptron
def Split(Data,labels,testSize, scale):
    
    
    x_train, x_test, y_train, y_test = train_test_split(Data, labels, test_size=testSize, random_state=42)
    scaler=StandardScaler()
    scaler.fit(x_train)
    x_train_scaled=scaler.transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    if scale:
        return x_train, x_test, y_train, y_test
    else:
        return  x_train_scaled, x_test_scaled, y_train, y_test
   

def Perceptron (Input0):
    
    NN0 = Dense(128, activation='sigmoid')(Input0)
    NN0 = Dense(64, activation='relu')(NN0)
    # NN0 = Dense(32, activation='relu')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    output = Dense(1, activation='linear')(NN0)
    
    model = Model(
        inputs=Input0,
        outputs=output)
    
    
    
    model.compile(optimizer=Adam(lr=.001),
                  loss='mean_squared_error',
                  metrics=['mape'])

    return model
    
def trainModel(model,x_train,y_train,plot):
    keras.backend.clear_session()
    history = model.fit(x_train, 
                        y_train, 
                        validation_split=0.2, 
                        batch_size=64,
                        epochs =1000,
                        verbose=1)
    
    if plot:
        
        plt.figure()
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
def evaluateModel(model,x_test,y_test):
    mse_neural, mape_neural = model.evaluate(x_test, y_test)
    print('Mean squared error from neural net: ', mse_neural)
    print('Mean absolute percentage error from neural net: ', mape_neural)
    predictions = model.predict(x_test).flatten()
    return predictions

def plotPredictions(predictions,y_test):
    plt.figure()
    plt.scatter(predictions,y_test)
    print(scipy.stats.pearsonr(predictions,y_test))
    lims=[0,100]
    plt.plot(lims,lims)
    plt.xlabel('predicted')
    plt.ylabel('ture values')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.show()