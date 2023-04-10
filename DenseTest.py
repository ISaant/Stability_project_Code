#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:41:45 2023

@author: sflores
"""

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from sklearn import metrics as Metrics

tf.keras.backend.clear_session()
drop=[i for i,x in enumerate(Par_Alpha.isna().any(axis=1)) if x]
Par_Alpha.drop(drop,inplace=True)
Par=np.array(Par_Alpha[Par_Alpha.columns[0:-1]])
Labels=np.array(Par_Alpha.loc[:,'Cohort'])
X0_train, X0_test, train_labels, test_labels = train_test_split(Par,
                                                                Labels,
                                                                test_size=.1)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

MLP0_Input=tf.keras.Input(shape=(X0_train.shape[1],), name="fRateT")
# NN0 = Dense(130, activation='sigmoid')(MLP0_Input)
# NN0 = Dense(512, activation='tanh')(MLP0_Input)
NN0 = Dense(32, activation='sigmoid')(MLP0_Input)
# NN0 = Dense(1024, activation='relu')(NN0)
# NN0 = Dense(1024, activation='tanh')(NN0)
# NN0 = Dense(1024, activation='sigmoid')(NN0)
# NN0 = Dense(1024, activation='tanh')(NN0)
# NN0 = Dense(128, activation='relu')(NN0)
# NN0 = Dense(64, activation='tanh')(NN0)
# NN0 = Dense(32, activation='sigmoid')(NN0)
# Prob_Dense = Dense(32, activation='relu')(NN0)
# # Prob_Dense = Dense(96, activation='tanh')(Prob_Dense)#Ampliar para arquitectura encoder decoder
# Prob_Dense = Dense(16, activation='tanh')(Prob_Dense)
output = Dense(2, activation='sigmoid',name='output')(NN0)

model = Model(inputs=MLP0_Input,
            outputs=output)
    
model.compile(optimizer=Adam(lr=.0001),
          loss=losses.binary_crossentropy,
          metrics=[metrics.binary_accuracy,metrics.Precision(), 
                    metrics.Recall(),
                    metrics.TrueNegatives(),
                    metrics.FalsePositives()])

train_labels=train_labels.tolist()
test_labels=test_labels.tolist()
# y_train =np.array(train_labels)
# y_test=np.array(test_labels)



history=model.fit(
    X0_train,
    y_train,
    epochs=1000,
    batch_size=256,
    validation_split=0.1,
    verbose=1)