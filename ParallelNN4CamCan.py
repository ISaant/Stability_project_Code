#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:40:15 2023

@author: sflores
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
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D,Conv1D, concatenate
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
from FunClassifier4CamCanJason import *

#%%
Data,labels=RemoveNan(restStateCropped, Age)
nPca=20
Data2=np.zeros((Data.shape[0],nPca,Data.shape[2]))
for i in range (68):
    Data2[:,:,i]=NNMatFac(Data[:,:,i],nPca)
x_train, x_test, y_train,y_test=Split(Data2,labels,.2)

#%%
tf.keras.backend.clear_session()
# Declare Inputs ==============================================================
Inputs=[]
for i in range (68):
    Inputs.append(tf.keras.Input(shape=(x_train.shape[1],), ))

#%% Define neurons per layer
l=[512,256,16]
#%% Create parallel networks 
#0
NN00 = Dense(l[0], activation='linear')(Inputs[0])
NN00 = Dense(l[1], activation='sigmoid')(NN00)
NN00 = Dense(l[2], activation='relu')(NN00)

#1
NN01 = Dense(l[0], activation='linear')(Inputs[1])
NN01 = Dense(l[1], activation='sigmoid')(NN01)
NN01 = Dense(l[2], activation='relu')(NN01)

#2
NN02 = Dense(l[0], activation='linear')(Inputs[2])
NN02 = Dense(l[1], activation='sigmoid')(NN02)
NN02 = Dense(l[2], activation='relu')(NN02)

#3
NN03 = Dense(l[0], activation='linear')(Inputs[3])
NN03 = Dense(l[1], activation='sigmoid')(NN03)
NN03 = Dense(l[2], activation='relu')(NN03)

#4
NN04 = Dense(l[0], activation='linear')(Inputs[4])
NN04 = Dense(l[1], activation='sigmoid')(NN04)
NN04 = Dense(l[2], activation='relu')(NN04)

#5
NN05 = Dense(l[0], activation='linear')(Inputs[5])
NN05 = Dense(l[1], activation='sigmoid')(NN05)
NN05 = Dense(l[2], activation='relu')(NN05)

#6
NN06 = Dense(l[0], activation='linear')(Inputs[6])
NN06 = Dense(l[1], activation='sigmoid')(NN06)
NN06 = Dense(l[2], activation='relu')(NN06)

#7
NN07 = Dense(l[0], activation='linear')(Inputs[7])
NN07 = Dense(l[1], activation='sigmoid')(NN07)
NN07 = Dense(l[2], activation='relu')(NN07)

#8
NN08 = Dense(l[0], activation='linear')(Inputs[8])
NN08 = Dense(l[1], activation='sigmoid')(NN08)
NN08 = Dense(l[2], activation='relu')(NN08)

#9
NN09 = Dense(l[0], activation='linear')(Inputs[9])
NN09 = Dense(l[1], activation='sigmoid')(NN09)
NN09 = Dense(l[2], activation='relu')(NN09)

#10
NN10 = Dense(l[0], activation='linear')(Inputs[10])
NN10 = Dense(l[1], activation='sigmoid')(NN10)
NN10 = Dense(l[2], activation='relu')(NN10)

#11
NN11 = Dense(l[0], activation='linear')(Inputs[11])
NN11 = Dense(l[1], activation='sigmoid')(NN11)
NN11 = Dense(l[2], activation='relu')(NN11)

#12
NN12 = Dense(l[0], activation='linear')(Inputs[12])
NN12 = Dense(l[1], activation='sigmoid')(NN12)
NN12 = Dense(l[2], activation='relu')(NN12)

#13
NN13 = Dense(l[0], activation='linear')(Inputs[13])
NN13 = Dense(l[1], activation='sigmoid')(NN13)
NN13 = Dense(l[2], activation='relu')(NN13)

#14
NN14 = Dense(l[0], activation='linear')(Inputs[14])
NN14 = Dense(l[1], activation='sigmoid')(NN14)
NN14 = Dense(l[2], activation='relu')(NN14)

#15
NN15 = Dense(l[0], activation='linear')(Inputs[15])
NN15 = Dense(l[1], activation='sigmoid')(NN15)
NN15 = Dense(l[2], activation='relu')(NN15)

#16
NN16 = Dense(l[0], activation='linear')(Inputs[16])
NN16 = Dense(l[1], activation='sigmoid')(NN16)
NN16 = Dense(l[2], activation='relu')(NN16)

#17
NN17 = Dense(l[0], activation='linear')(Inputs[17])
NN17 = Dense(l[1], activation='sigmoid')(NN17)
NN17 = Dense(l[2], activation='relu')(NN17)

#18
NN18 = Dense(l[0], activation='linear')(Inputs[18])
NN18 = Dense(l[1], activation='sigmoid')(NN18)
NN18 = Dense(l[2], activation='relu')(NN18)

#19
NN19 = Dense(l[0], activation='linear')(Inputs[19])
NN19 = Dense(l[1], activation='sigmoid')(NN19)
NN19 = Dense(l[2], activation='relu')(NN19)

#20
NN20 = Dense(l[0], activation='linear')(Inputs[20])
NN20 = Dense(l[1], activation='sigmoid')(NN20)
NN20 = Dense(l[2], activation='relu')(NN20)

#21
NN21 = Dense(l[0], activation='linear')(Inputs[21])
NN21 = Dense(l[1], activation='sigmoid')(NN21)
NN21 = Dense(l[2], activation='relu')(NN21)

#22
NN22 = Dense(l[0], activation='linear')(Inputs[22])
NN22 = Dense(l[1], activation='sigmoid')(NN22)
NN22 = Dense(l[2], activation='relu')(NN22)

#23
NN23 = Dense(l[0], activation='linear')(Inputs[23])
NN23 = Dense(l[1], activation='sigmoid')(NN23)
NN23 = Dense(l[2], activation='relu')(NN23)

#24
NN24 = Dense(l[0], activation='linear')(Inputs[24])
NN24 = Dense(l[1], activation='sigmoid')(NN24)
NN24 = Dense(l[2], activation='relu')(NN24)

#25
NN25 = Dense(l[0], activation='linear')(Inputs[25])
NN25 = Dense(l[1], activation='sigmoid')(NN25)
NN25 = Dense(l[2], activation='relu')(NN25)

#26
NN26 = Dense(l[0], activation='linear')(Inputs[26])
NN26 = Dense(l[1], activation='sigmoid')(NN26)
NN26 = Dense(l[2], activation='relu')(NN26)

#27
NN27 = Dense(l[0], activation='linear')(Inputs[27])
NN27 = Dense(l[1], activation='sigmoid')(NN27)
NN27 = Dense(l[2], activation='relu')(NN27)

#28
NN28 = Dense(l[0], activation='linear')(Inputs[28])
NN28 = Dense(l[1], activation='sigmoid')(NN28)
NN28 = Dense(l[2], activation='relu')(NN28)

#29
NN29 = Dense(l[0], activation='linear')(Inputs[29])
NN29 = Dense(l[1], activation='sigmoid')(NN29)
NN29 = Dense(l[2], activation='relu')(NN29)

#30
NN30 = Dense(l[0], activation='linear')(Inputs[30])
NN30 = Dense(l[1], activation='sigmoid')(NN30)
NN30 = Dense(l[2], activation='relu')(NN30)

#31
NN31 = Dense(l[0], activation='linear')(Inputs[31])
NN31 = Dense(l[1], activation='sigmoid')(NN31)
NN31 = Dense(l[2], activation='relu')(NN31)

#32
NN32 = Dense(l[0], activation='linear')(Inputs[32])
NN32 = Dense(l[1], activation='sigmoid')(NN32)
NN32 = Dense(l[2], activation='relu')(NN32)

#33
NN33 = Dense(l[0], activation='linear')(Inputs[33])
NN33 = Dense(l[1], activation='sigmoid')(NN33)
NN33 = Dense(l[2], activation='relu')(NN33)

#34
NN34 = Dense(l[0], activation='linear')(Inputs[34])
NN34 = Dense(l[1], activation='sigmoid')(NN34)
NN34 = Dense(l[2], activation='relu')(NN34)

#35
NN35 = Dense(l[0], activation='linear')(Inputs[35])
NN35 = Dense(l[1], activation='sigmoid')(NN35)
NN35 = Dense(l[2], activation='relu')(NN35)

#36
NN36 = Dense(l[0], activation='linear')(Inputs[36])
NN36 = Dense(l[1], activation='sigmoid')(NN36)
NN36 = Dense(l[2], activation='relu')(NN36)

#37
NN37 = Dense(l[0], activation='linear')(Inputs[37])
NN37 = Dense(l[1], activation='sigmoid')(NN37)
NN37 = Dense(l[2], activation='relu')(NN37)

#38
NN38 = Dense(l[0], activation='linear')(Inputs[38])
NN38 = Dense(l[1], activation='sigmoid')(NN38)
NN38 = Dense(l[2], activation='relu')(NN38)

#39
NN39 = Dense(l[0], activation='linear')(Inputs[39])
NN39 = Dense(l[1], activation='sigmoid')(NN39)
NN39 = Dense(l[2], activation='relu')(NN39)

#40
NN40 = Dense(l[0], activation='linear')(Inputs[40])
NN40 = Dense(l[1], activation='sigmoid')(NN40)
NN40 = Dense(l[2], activation='relu')(NN40)

#41
NN41 = Dense(l[0], activation='linear')(Inputs[41])
NN41 = Dense(l[1], activation='sigmoid')(NN41)
NN41 = Dense(l[2], activation='relu')(NN41)

#42
NN42 = Dense(l[0], activation='linear')(Inputs[42])
NN42 = Dense(l[1], activation='sigmoid')(NN42)
NN42 = Dense(l[2], activation='relu')(NN42)

#43
NN43 = Dense(l[0], activation='linear')(Inputs[43])
NN43 = Dense(l[1], activation='sigmoid')(NN43)
NN43 = Dense(l[2], activation='relu')(NN43)

#44
NN44 = Dense(l[0], activation='linear')(Inputs[44])
NN44 = Dense(l[1], activation='sigmoid')(NN44)
NN44 = Dense(l[2], activation='relu')(NN44)

#45
NN45 = Dense(l[0], activation='linear')(Inputs[45])
NN45 = Dense(l[1], activation='sigmoid')(NN45)
NN45 = Dense(l[2], activation='relu')(NN45)

#46
NN46 = Dense(l[0], activation='linear')(Inputs[46])
NN46 = Dense(l[1], activation='sigmoid')(NN46)
NN46 = Dense(l[2], activation='relu')(NN46)

#47
NN47 = Dense(l[0], activation='linear')(Inputs[47])
NN47 = Dense(l[1], activation='sigmoid')(NN47)
NN47 = Dense(l[2], activation='relu')(NN47)

#48
NN48 = Dense(l[0], activation='linear')(Inputs[48])
NN48 = Dense(l[1], activation='sigmoid')(NN48)
NN48 = Dense(l[2], activation='relu')(NN48)

#49
NN49 = Dense(l[0], activation='linear')(Inputs[49])
NN49 = Dense(l[1], activation='sigmoid')(NN49)
NN49 = Dense(l[2], activation='relu')(NN49)

#50
NN50 = Dense(l[0], activation='linear')(Inputs[50])
NN50 = Dense(l[1], activation='sigmoid')(NN50)
NN50 = Dense(l[2], activation='relu')(NN50)

#51
NN51 = Dense(l[0], activation='linear')(Inputs[51])
NN51 = Dense(l[1], activation='sigmoid')(NN51)
NN51 = Dense(l[2], activation='relu')(NN51)

#52
NN52 = Dense(l[0], activation='linear')(Inputs[52])
NN52 = Dense(l[1], activation='sigmoid')(NN52)
NN52 = Dense(l[2], activation='relu')(NN52)

#53
NN53 = Dense(l[0], activation='linear')(Inputs[53])
NN53 = Dense(l[1], activation='sigmoid')(NN53)
NN53 = Dense(l[2], activation='relu')(NN53)

#54
NN54 = Dense(l[0], activation='linear')(Inputs[54])
NN54 = Dense(l[1], activation='sigmoid')(NN54)
NN54 = Dense(l[2], activation='relu')(NN54)

#55
NN55 = Dense(l[0], activation='linear')(Inputs[55])
NN55 = Dense(l[1], activation='sigmoid')(NN55)
NN55 = Dense(l[2], activation='relu')(NN55)

#56
NN56 = Dense(l[0], activation='linear')(Inputs[56])
NN56 = Dense(l[1], activation='sigmoid')(NN56)
NN56 = Dense(l[2], activation='relu')(NN56)

#57
NN57 = Dense(l[0], activation='linear')(Inputs[57])
NN57 = Dense(l[1], activation='sigmoid')(NN57)
NN57 = Dense(l[2], activation='relu')(NN57)

#58
NN58 = Dense(l[0], activation='linear')(Inputs[58])
NN58 = Dense(l[1], activation='sigmoid')(NN58)
NN58 = Dense(l[2], activation='relu')(NN58)

#59
NN59 = Dense(l[0], activation='linear')(Inputs[59])
NN59 = Dense(l[1], activation='sigmoid')(NN59)
NN59 = Dense(l[2], activation='relu')(NN59)

#60
NN60 = Dense(l[0], activation='linear')(Inputs[60])
NN60 = Dense(l[1], activation='sigmoid')(NN60)
NN60 = Dense(l[2], activation='relu')(NN60)

#61
NN61 = Dense(l[0], activation='linear')(Inputs[61])
NN61 = Dense(l[1], activation='sigmoid')(NN61)
NN61 = Dense(l[2], activation='relu')(NN61)

#62
NN62 = Dense(l[0], activation='linear')(Inputs[62])
NN62 = Dense(l[1], activation='sigmoid')(NN62)
NN62 = Dense(l[2], activation='relu')(NN62)

#63
NN63 = Dense(l[0], activation='linear')(Inputs[63])
NN63 = Dense(l[1], activation='sigmoid')(NN63)
NN63 = Dense(l[2], activation='relu')(NN63)

#64
NN64 = Dense(l[0], activation='linear')(Inputs[64])
NN64 = Dense(l[1], activation='sigmoid')(NN64)
NN64 = Dense(l[2], activation='relu')(NN64)

#65
NN65 = Dense(l[0], activation='linear')(Inputs[65])
NN65 = Dense(l[1], activation='sigmoid')(NN65)
NN65 = Dense(l[2], activation='relu')(NN65)

#66
NN66 = Dense(l[0], activation='linear')(Inputs[66])
NN66 = Dense(l[1], activation='sigmoid')(NN66)
NN66 = Dense(l[2], activation='relu')(NN66)

#67
NN67 = Dense(l[0], activation='linear')(Inputs[67])
NN67 = Dense(l[1], activation='sigmoid')(NN67)
NN67 = Dense(l[2], activation='relu')(NN67)



#%% Concatenate outputs and design last nn
feature_layers=concatenate([NN00, NN01, NN02, NN03, NN04, NN05, NN06, NN07, NN08, NN09,
                            NN10, NN11, NN12, NN13, NN14, NN15, NN16, NN17, NN18, NN19,
                            NN20, NN21, NN22, NN23, NN24, NN25, NN26, NN27, NN28, NN29,
                            NN30, NN31, NN32, NN33, NN34, NN35, NN36, NN37, NN38, NN39,
                            NN40, NN41, NN42, NN43, NN44, NN45, NN46, NN47, NN48, NN49,
                            NN50, NN51, NN52, NN53, NN54, NN55, NN56, NN57, NN58, NN59,
                            NN60, NN61, NN62, NN63, NN64, NN65, NN66, NN67])
NN = Dropout(.3)(feature_layers)
NN = Dense(l[0], activation='linear')(NN)
NN = Dense(l[1], activation='sigmoid')(NN)
NN = Dense(l[2], activation='relu')(NN)
output = Dense(1, activation='linear')(NN)

#%% Define Inputs, Outputs, loss and metrics 
loss='mean_squared_error',
metrics=['mape']
# model = Model(
#     inputs=[Inputs[0], Inputs[1], Inputs[2], Inputs[3], Inputs[4], Inputs[5], Inputs[6], Inputs[7],Inputs[8], Inputs[9],
#             Inputs[10], Inputs[11], Inputs[12], Inputs[13], Inputs[14], Inputs[15], Inputs[16], Inputs[17],Inputs[18], Inputs[19],
#             Inputs[20], Inputs[21], Inputs[22], Inputs[23], Inputs[24], Inputs[25], Inputs[26], Inputs[27],Inputs[28], Inputs[29],
#             Inputs[30], Inputs[31], Inputs[32], Inputs[33], Inputs[34], Inputs[35], Inputs[36], Inputs[37],Inputs[38], Inputs[39],
#             Inputs[40], Inputs[41], Inputs[42], Inputs[43], Inputs[44], Inputs[45], Inputs[46], Inputs[47],Inputs[48], Inputs[49],
#             Inputs[50], Inputs[51], Inputs[52], Inputs[53], Inputs[54], Inputs[55], Inputs[56], Inputs[57],Inputs[58], Inputs[59],
#             Inputs[60], Inputs[61], Inputs[62], Inputs[63], Inputs[64], Inputs[65], Inputs[66], Inputs[67]],
#     outputs=output)
model = Model(
    inputs=Inputs,
    outputs=output)


# print(model.summary())
model.compile(optimizer=Adam(learning_rate=.001),
              loss=loss,
              metrics=metrics)
tf.keras.backend.clear_session()
trainModel(model,[x_train[:,:,0],x_train[:,:,1],x_train[:,:,2],x_train[:,:,3],x_train[:,:,4],x_train[:,:,5],x_train[:,:,6],x_train[:,:,7],x_train[:,:,8],x_train[:,:,9],
 x_train[:,:,10],x_train[:,:,11],x_train[:,:,12],x_train[:,:,13],x_train[:,:,14],x_train[:,:,15],x_train[:,:,16],x_train[:,:,17],x_train[:,:,18],x_train[:,:,19],
 x_train[:,:,20],x_train[:,:,21],x_train[:,:,22],x_train[:,:,23],x_train[:,:,24],x_train[:,:,25],x_train[:,:,26],x_train[:,:,27],x_train[:,:,28],x_train[:,:,29],
 x_train[:,:,30],x_train[:,:,31],x_train[:,:,32],x_train[:,:,33],x_train[:,:,34],x_train[:,:,35],x_train[:,:,36],x_train[:,:,37],x_train[:,:,38],x_train[:,:,39],
 x_train[:,:,40],x_train[:,:,41],x_train[:,:,42],x_train[:,:,43],x_train[:,:,44],x_train[:,:,45],x_train[:,:,46],x_train[:,:,47],x_train[:,:,48],x_train[:,:,49],
 x_train[:,:,50],x_train[:,:,51],x_train[:,:,52],x_train[:,:,53],x_train[:,:,54],x_train[:,:,55],x_train[:,:,56],x_train[:,:,57],x_train[:,:,58],x_train[:,:,59],
 x_train[:,:,60],x_train[:,:,61],x_train[:,:,62],x_train[:,:,63],x_train[:,:,64],x_train[:,:,65],x_train[:,:,66],x_train[:,:,67]],y_train,50,True)
# predNN=evaluateRegModel(model,x_test,y_test)
predNN = model.predict([x_test[:,:,0],x_test[:,:,1],x_test[:,:,2],x_test[:,:,3],x_test[:,:,4],x_test[:,:,5],x_test[:,:,6],x_test[:,:,7],x_test[:,:,8],x_test[:,:,9],
 x_test[:,:,10],x_test[:,:,11],x_test[:,:,12],x_test[:,:,13],x_test[:,:,14],x_test[:,:,15],x_test[:,:,16],x_test[:,:,17],x_test[:,:,18],x_test[:,:,19],
 x_test[:,:,20],x_test[:,:,21],x_test[:,:,22],x_test[:,:,23],x_test[:,:,24],x_test[:,:,25],x_test[:,:,26],x_test[:,:,27],x_test[:,:,28],x_test[:,:,29],
 x_test[:,:,30],x_test[:,:,31],x_test[:,:,32],x_test[:,:,33],x_test[:,:,34],x_test[:,:,35],x_test[:,:,36],x_test[:,:,37],x_test[:,:,38],x_test[:,:,39],
 x_test[:,:,40],x_test[:,:,41],x_test[:,:,42],x_test[:,:,43],x_test[:,:,44],x_test[:,:,45],x_test[:,:,46],x_test[:,:,47],x_test[:,:,48],x_test[:,:,49],
 x_test[:,:,50],x_test[:,:,51],x_test[:,:,52],x_test[:,:,53],x_test[:,:,54],x_test[:,:,55],x_test[:,:,56],x_test[:,:,57],x_test[:,:,58],x_test[:,:,59],
 x_test[:,:,60],x_test[:,:,61],x_test[:,:,62],x_test[:,:,63],x_test[:,:,64],x_test[:,:,65],x_test[:,:,66],x_test[:,:,67]])
NNPred=plotPredictionsReg(predNN,y_test)
print(NNPred)
