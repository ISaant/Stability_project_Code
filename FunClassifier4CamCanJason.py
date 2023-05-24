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

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D,Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler

#%% Scale data
def Scale(Data):
    
    scaler=StandardScaler()
    scaler.fit(Data)
    Data=scaler.transform(Data)
    return Data

#%% Split data
def Split(Data,labels,testSize):
    x_train, x_test, y_train, y_test = train_test_split(Data, labels, test_size=testSize)
    return  x_train, x_test, y_train, y_test
   
#%% Perceptron
def Perceptron (Input0,classification):
    # print(classification)
    tf.keras.backend.clear_session()
    NN0 = Dense(2048, activation='linear')(Input0)
    NN0 = Dense(256, activation='sigmoid')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    output = Dense(1, activation='linear')(NN0)
    loss='mean_squared_error',
    metrics=['mape']
    if classification:
        output = Dense(7, activation='softmax')(NN0)
        loss='categorical_crossentropy',
        metrics=[Precision(),Recall()]
    model = Model(
        inputs=Input0,
        outputs=output)
    
    
    # print(model.summary())
    model.compile(optimizer=Adam(learning_rate=.001),
                  loss=loss,
                  metrics=metrics)

    return model

#%% CNN1D
def CCN1D (Input0):
    
    
   
    CNN0 = Conv1D(128, 3, activation="relu",padding="same")(Input0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = Conv1D(64, 3, activation="relu",padding="same")(CNN0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = MaxPooling1D(pool_size=2)(CNN0)
    CNN0 = Conv1D(32, 3, activation="relu",padding="same")(CNN0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = Conv1D(16, 3, activation="relu",padding="same")(CNN0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = Dropout(0.1)(CNN0)
    CNN0 = MaxPooling1D(pool_size=2)(CNN0)
    Flat = Flatten()(CNN0)
    
    cnn = Model(Input0, Flat)
    cnn.summary()

    NN0 = Dense(512, activation='relu')(Flat)
    NN0 = Dense(128, activation='sigmoid')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    # NN0 = Dense(32, activation='relu')(NN0)
    NN1 = Dense(16, activation='relu')(NN0)
    output = Dense(1, activation='linear')(NN0)

    model = Model(inputs=Input0,
                outputs=output)
    
    model.compile(optimizer=Adam(lr=.0001),
                  loss='mean_squared_error',
                  metrics=['mape'])

    return model

#%%
def trainModel(model,x_train,y_train,epochs,plot):
    keras.backend.clear_session()
    history = model.fit(x_train, 
                        y_train, 
                        validation_split=0.2, 
                        batch_size=64,
                        epochs =epochs,
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
    
def evaluateRegModel(model,x_test,y_test):
    mse_neural, mape_neural = model.evaluate(x_test, y_test, verbose=0)
    # print('Mean squared error from neural net: ', mse_neural)
    # print('Mean absolute percentage error from neural net: ', mape_neural)
    predictions = model.predict(x_test).flatten()
    return predictions

def evaluateClassModel(model,x_test,y_test):
    # print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    print(results)

#%% Lasso





#%% Function to plot predictions

def plotPredictionsReg(predictions,y_test):
    plt.figure()
    plt.scatter(predictions,y_test)
    pearson=scipy.stats.pearsonr(predictions,y_test)
    # print(pearson)
    lims=[0,100]
    plt.plot(lims,lims)
    plt.xlabel('predicted')
    plt.ylabel('ture values')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.show()
    return pearson[0]