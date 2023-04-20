#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:02:06 2023

@author: sflores
"""

import numpy as np
import matplotlib.pyplot as plt
from Fun3 import *
from FunClassifier import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
#%%
APer_Alpha_1peak = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=True, in_between=[1,50],
                      max_n_peaks=1, fit='fixed')

#%%
freqs=np.linspace(1,49,(49*3),endpoint=True)
PSD=np.array(APer_Alpha_1peak.whitened)
plt.figure()
meanA=np.mean(PSD[:int(PSD.shape[0]/2)],axis=0)
meanB=np.mean(PSD[int(PSD.shape[0]/2):],axis=0)
stdA=np.std(PSD[:int(PSD.shape[0]/2)],axis=0)
stdB=np.std(PSD[int(PSD.shape[0]/2):],axis=0)
# plt.plot(np.log(freqs),np.log(meanA),'r')
# plt.plot(np.log(freqs),np.log(meanB),'g')
print(np.sqrt(PSD.shape[0]/2))
coefIntervalA=(1.96*(stdA)/np.sqrt(PSD.shape[0]/2))
coefIntervalB=(1.96*(stdB)/np.sqrt(PSD.shape[0]/2))
# plt.fill_between(np.log(freqs),np.log(meanA+coefIntervalA),np.log(meanA-coefIntervalA),alpha=.5,color='r')
# plt.fill_between(np.log(freqs),np.log(meanB+coefIntervalB),np.log(meanB-coefIntervalB),alpha=.5,color='g')
plt.plot(freqs,meanA,'r')
plt.plot(freqs,meanB,'g')
plt.fill_between(freqs,meanA+coefIntervalA,meanA-coefIntervalA,alpha=.5,color='r')
plt.fill_between(freqs,meanB+coefIntervalB,meanB-coefIntervalB,alpha=.5,color='g')

plt.title('Band= '+Generate_Data.band+', Mean A vs B groups, all widows, all sujects. Freqs between'+str(APer_Alpha_1peak.in_between))
plt.show()

#%%
labels=APer_Alpha_1peak.periodic['Cohort'].to_numpy()
matrix=APer_Alpha_1peak.whitened.to_numpy()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_trained_samples=scaler.fit_transform(matrix)
PSD = matrix.reshape(matrix.shape[0], matrix.shape[1], 1)

x_train, x_test, y_train, y_test = train_test_split(PSD,
                                                    labels,
                                                    test_size=.3)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
tf.keras.backend.clear_session()
model = Sequential ([
    Conv1D(128,25,activation='relu',input_shape=(x_train.shape[1],1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, strides=1),
    Conv1D(256,15,activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, strides=1),
    Conv1D(128,5,activation='relu'),
    BatchNormalization(),
    # GlobalMaxPooling1D(),
    Flatten(),
    # Dense(512,activation='tanh'),
    # Dense(128,activation='relu'),
    # Dropout(.1),
    Dense(32,activation='tanh'),
    # Dropout(.1),
    Dense(64,activation='relu'),
    # Dropout(.3),
    Dense(2,activation='softmax')
    
    ])

print(model.summary())
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=[Accuracy(),Precision(),Recall()])

history=model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.1,
    verbose=1)

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=64)
print(results)
# return results