#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:41:55 2023

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

#%%
def nn_for_coefs(coeffs):
    
    
    labels=coeffs['Cohort'].to_numpy()
    matrix=coeffs.iloc[:,2:-1].to_numpy()
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_trained_samples=scaler.fit_transform(matrix)
    
    X0_train, X0_test, y_train, y_test = train_test_split(scaled_trained_samples,
                                                                    labels,
                                                                    test_size=.3)
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    tf.keras.backend.clear_session()
    model = Sequential ([
        Dense(32,input_shape=(X0_train.shape[1],),activation='linear'),
        # Dense(64,activation='relu'),
        # Dense(16,activation='tanh'),
        # Dense(8,activation='relu'),
        Dense(2,activation='softmax')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
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

#%%
def nn_for_PwsAlphaBeta(coeffs):
    
    
    labels=coeffs['Cohort'].to_numpy()
    matrix=coeffs.iloc[:,:-1].to_numpy()
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_trained_samples=scaler.fit_transform(matrix)
    
    X0_train, X0_test, y_train, y_test = train_test_split(scaled_trained_samples,
                                                                    labels,
                                                                    test_size=.3)
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    tf.keras.backend.clear_session()
    model = Sequential ([
        Dense(32,input_shape=(X0_train.shape[1],),activation='sigmoid'),
        # Dense(64,activation='relu'),
        # Dense(16,activation='tanh'),
        Dense(8,activation='relu'),
        Dense(2,activation='softmax')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
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

def nn_filterbank(filterbank,labels):

    
    X0_train, X0_test, y_train, y_test = train_test_split(filterbank,
                                                          labels,
                                                          test_size=.3)
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    tf.keras.backend.clear_session()
    model = Sequential ([
        Dense(64,input_shape=(X0_train.shape[1],),activation='sigmoid'),
        # Dense(128,activation='relu'),
        # Dense(128,activation='tanh'),
        # Dense(64,activation='relu'),
        Dense(32,activation='relu'),
        Dense(16,activation='tanh'),
        Dense(8,activation='linear'),
        Dense(2,activation='softmax')
        ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
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

#%%
def SVM(coeffs):
    X=coeffs.iloc[:,:2].to_numpy()
    y=coeffs.iloc[:,2].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                          test_size=.3)
    C = 1.0  # SVM regularization parameter
    clf = svm.LinearSVC(C=C, max_iter=1000).fit(x_train, y_train)
        

    title ="SVC with linear kernel"
        

    # Set-up 2x2 grid for plotting.
    # fig, ax = plt.subplots(1, 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    y_pred=clf.predict(x_test)
    acc=accuracy_score(y_test, y_pred)*100
    print(acc)
    # disp = DecisionBoundaryDisplay.from_estimator(
    #     clf,
    #     X,
    #     response_method="predict",
    #     cmap=plt.cm.coolwarm,
    #     alpha=0.8,
    #     ax=ax,
    #     xlabel='Alpha_Power',
    #     ylabel='Beta_Power',
    # )
    # ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax.set_title(title+', acc= '+str((accuracy_score(y_test, y_pred)*100)))

    plt.show()
    return acc