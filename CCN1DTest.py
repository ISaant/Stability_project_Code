#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:18:35 2023

@author: sflores
"""
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


PSD=np.array(DataFrame[DataFrame.columns[0:-2]])
freqs=np.arange(1,40,.3)
meanA=np.mean(PSD[:250],axis=0)
meanB=np.mean(PSD[250:],axis=0)
stdA=np.std(PSD[:250],axis=0)
stdB=np.std(PSD[250:],axis=0)
plt.plot(freqs,meanA,'r')
plt.plot(freqs,meanB,'g')
plt.fill_between(freqs,meanA+stdA,meanA-stdA,alpha=.5,color='r')
plt.fill_between(freqs,meanB+stdB,meanB-stdB,alpha=.5,color='g')

PSD = PSD.reshape(PSD.shape[0], PSD.shape[1], 1)
Labels=np.array(DataFrame[DataFrame.columns[-1]])
x_train, x_test, train_labels, test_labels = train_test_split(PSD,
                                                                Labels,
                                                                test_size=.1)

# y_train = to_categorical(train_labels)
# y_test = to_categorical(test_labels)
y_train = train_labels
y_test = test_labels

tf.keras.backend.clear_session()

CNN1D_Input=tf.keras.Input(shape=(x_train.shape[1],1), name="fRateT")
CNN0 = Conv1D(128, 7, activation="relu",padding="same")(CNN1D_Input)
CNN0 = BatchNormalization() (CNN0)
CNN0 = Conv1D(64, 7, activation="relu",padding="same")(CNN0)
CNN0 = BatchNormalization() (CNN0)
CNN0 = MaxPooling1D(pool_size=2)(CNN0)
CNN0 = Conv1D(32, 3, activation="relu",padding="same")(CNN0)
CNN0 = BatchNormalization() (CNN0)
CNN0 = Conv1D(16, 3, activation="relu",padding="same")(CNN0)
CNN0 = BatchNormalization() (CNN0)
CNN0 = Dropout(0.1)(CNN0)
CNN0 = MaxPooling1D(pool_size=2)(CNN0)
Flat = Flatten()(CNN0)

cnn = Model(CNN1D_Input, Flat)
cnn.summary()

NN0 = Dense(512, activation='relu')(Flat)
NN0 = Dense(1024, activation='tanh')(NN0)
NN0 = Dense(512, activation='relu')(NN0)
NN0 = Dense(256, activation='tanh')(NN0)
NN0 = Dense(128, activation='sigmoid')(NN0)
NN0 = Dense(64, activation='relu')(NN0)
NN0 = Dense(256, activation='tanh')(NN0)#Ampliar para arquitectura encoder decoder
NN0 = Dense(32, activation='relu')(NN0)
output = Dense(1, activation='sigmoid',name='output')(NN0)

model = Model(inputs=CNN1D_Input,
            outputs=output)

# model.summary()
    


#%%

model.compile(optimizer=Adam(learning_rate=.0001),
          # loss=losses.binary_crossentropy,
          loss="mse",
          metrics=[metrics.binary_accuracy,metrics.Precision(), 
                    metrics.Recall()])

#%%
history=model.fit(
    x_train,
    y_train,
    epochs=1000,
    # batch_size=int(len(x_train)/5),
    batch_size=512,
    validation_split=0.1,
    verbose=1)

#%%


#%% 
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=256)
print("test loss, test acc:", results)
