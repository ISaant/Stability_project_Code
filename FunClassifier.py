#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:41:55 2023

@author: isaac
"""

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall

physical_devices= tf.config.experimental.list_physical_devices('GPU')
print ("Num GPUs Available: ",len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0],True)
def nn_for_coefs(coeffs):
    
    
    labels=coeffs['Cohort'].to_numpy()
    matrix=coeffs.iloc[:,2:-1].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_trained_samples=scaler.fit_transform(matrix)
    
    X0_train, X0_test, y_train, y_test = train_test_split(scaled_trained_samples,
                                                                    labels,
                                                                    test_size=.3)
    tf.keras.backend.clear_session()
    model = Sequential ([
        Dense(32,input_shape=(X0_train.shape[1],),activation='sigmoid'),
        Dense(64,activation='relu'),
        Dense(16,activation='tanh'),
        Dense(8,activation='relu'),
        Dense(1,activation='sigmoid')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=[Accuracy(),Precision(),Recall()])
    
    history=model.fit(
        X0_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_split=0.1,
        verbose=0)
    
    print("Evaluate on test data")
    results = model.evaluate(X0_test, y_test, batch_size=128)
    print(results)
    return results

def nn_for_PwsAlphaBeta(coeffs):
    
    
    labels=coeffs['Cohort'].to_numpy()
    matrix=coeffs.iloc[:,:-1].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_trained_samples=scaler.fit_transform(matrix)
    
    X0_train, X0_test, y_train, y_test = train_test_split(scaled_trained_samples,
                                                                    labels,
                                                                    test_size=.1)
    tf.keras.backend.clear_session()
    model = Sequential ([
        Dense(32,input_shape=(X0_train.shape[1],),activation='sigmoid'),
        Dense(64,activation='relu'),
        Dense(16,activation='tanh'),
        Dense(8,activation='relu'),
        Dense(1,activation='sigmoid')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=[Accuracy(),Precision(),Recall()])
    
    history=model.fit(
        X0_train,
        y_train,
        epochs=1000,
        batch_size=16,
        validation_split=0.1,
        verbose=0)
    
    print("Evaluate on test data")
    results = model.evaluate(X0_test, y_test, batch_size=128)
    print(results)
    return results

def nn_filterbank(filterbank,labels):

    
    X0_train, X0_test, y_train, y_test = train_test_split(filterbank,
                                                          labels,
                                                          test_size=.1)
    tf.keras.backend.clear_session()
    model = Sequential ([
        Dense(64,input_shape=(X0_train.shape[1],),activation='sigmoid'),
        # Dense(128,activation='relu'),
        # Dense(128,activation='tanh'),
        # Dense(64,activation='relu'),
        # Dense(128,activation='relu'),
        Dense(16,activation='tanh'),
        Dense(8,activation='linear'),
        Dense(1,activation='sigmoid')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=[Accuracy(),Precision(),Recall()])
    
    history=model.fit(
        X0_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_split=0.1,
        verbose=0)
    print("Evaluate on test data")
    results = model.evaluate(X0_test, y_test, batch_size=128)
    print(results)
    return results