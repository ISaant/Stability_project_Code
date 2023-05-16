#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:03:35 2023

@author: isaac
"""
import os
import pandas as pd
import numpy as np
import copy
import seaborn as sns
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from Fun4CamCanJason import *
from fooof import FOOOF
from sklearn.model_selection import train_test_split
#%% Hyperparameters

#fooof 
inBetween=[1,40]

#PSD
freqs=np.arange(0,150,.5)
alfaBetaFreqs=[0,50]
columns= [i for i, x in enumerate((freqs>=alfaBetaFreqs[0]) & (freqs<alfaBetaFreqs[1])) if x]
#columns is used to select the region of the PSD we are interested in

#%% Directories
current_path = os.getcwd()
parentPath=os.path.abspath(os.path.join(current_path,os.pardir))
path2Data=parentPath+'/Stability-project_db/CAMCAN_Jason_PrePro/'
mainDir=np.sort(os.listdir(path2Data))
restStateDir=np.sort(os.listdir(path2Data+mainDir[1]+'/'))
taskDir=np.sort(os.listdir(path2Data+mainDir[2]+'/'))

#%% Read demografics and average Resting state using all time windows

demographics=pd.read_csv(path2Data+mainDir[0])
Catell=demographics['Catell_score'].to_numpy()
Age=demographics['age'].to_numpy()
for e,file in enumerate(tqdm(restStateDir)):
    matrix=pd.read_csv(path2Data+mainDir[1]+'/'+file,header=None)
    if e == 0:
        print (e)
        restState=matrix
        continue
    restState+=matrix
restStateOriginal=restState
restStateOriginal/=(e+1)
restState = myReshape(restStateOriginal.to_numpy()) #reshape into [ROI,subjects,PSD]
restStateCropped = restState[:,columns,:] # Select the band-width of interest

#%% Plot global mean and mean per ROI
psdPlot(freqs[columns], restStateCropped)

#%%  FOOOF

periodic, aperiodic, whitened, parameters, freqsInBetween=fooof(restStateCropped, freqs[columns], inBetween)


#%% Plot global mean and mean per ROI after FOOOF
psdPlot(freqsInBetween, periodic)
psdPlot(freqsInBetween, aperiodic)
psdPlot(freqsInBetween, whitened)

# #%% PCA statistics
# Data=whitened #change this in order to test diferent psd (original, per, aper, whitened, etc )
# Sub,PSD,ROI=Data.shape
# nPca=25
# PCA=np.zeros((Sub,nPca+2,ROI))
# Var=[]
# Pro=[]
# for roi in  tqdm(range(ROI)):
#     df=pd.DataFrame(Data[:,:,roi])
#     proyecciones,pro2use,prop_varianza_acum=ACP (df,False,False,25)
#     Var.append(prop_varianza_acum)
#     PCA[:,:nPca,roi]=pro2use
#     for sub in range (Sub):  
#         PCA[sub,nPca,roi]=parameters[0][sub][0]
#         PCA[sub,nPca+1,roi]=parameters[0][sub][1]
# Var=pd.DataFrame(Var)
# sns.boxplot(Var)
# plt.ylim((0,1))

#%% myPCA statistics
figure()
Data=whitened #change this in order to test diferent psd (original, per, aper, whitened, etc )
Sub,PSD,ROI=Data.shape
nPca=25
PCA=np.zeros((Sub,nPca+2,ROI))
Var=[]
Pro=[]
for roi in  tqdm(range(ROI)):
    df=pd.DataFrame(Data[:,:,roi])
    pca_df,pro2use,prop_varianza_acum=myPCA(df,False,nPca)
    Var.append(prop_varianza_acum)
    PCA[:,:nPca,roi]=pro2use
    for sub in range (Sub):  
        PCA[sub,nPca,roi]=parameters[0][sub][0]
        PCA[sub,nPca+1,roi]=parameters[0][sub][1]
Var=pd.DataFrame(Var)
sns.boxplot(Var)
plt.ylim((0,1))

#%%
pca_df,pro2use,prop_varianza_acum=myPCA (restStateOriginal,True, nPca)


#%% Delete nan from target drop same subject
target=Age
idx=np.argwhere(np.isnan(target))
labels=np.delete(target, idx)
Data=np.delete(whitened, idx,axis=0)
#%%
# Data=Data.reshape((603,27,68))
keras.backend.clear_session()
x_train, x_test, y_train, y_test = train_test_split(Data[:,:100], labels, test_size=0.3, random_state=42)
scaler=StandardScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
Input0=tf.keras.Input(shape=(x_train_scaled.shape[1],), )
# Input1=tf.keras.Input(shape=(x_train[:,1].shape[1],), )

NN0 = Dense(128, activation='sigmoid')(Input0)
NN0 = Dense(64, activation='relu')(NN0)
# NN0 = Dense(32, activation='sigmoid')(NN0)
NN0 = Dense(32, activation='sigmoid')(NN0)
NN0 = Dense(16, activation='relu')(NN0)
output = Dense(1)(NN0)
# NN1 = Dense(128, activation='sigmoid')(Input1)
# NN1 = Dense(64, activation='selu')(NN1)
# NN1 = Dense(32, activation='tanh')(NN1)
# NN1 = Dense(32, activation='relu')(NN1)

# x = layers.concatenate([NN0,NN1])
# x = layers.concatenate([NN,CNN])

# Prob_Dense = Dense(64, activation='tanh',name="Last_NN_Targets")(x)
# # Prob_Dense = Dropout(.3)(Prob_Dense)
# Prob_Dense = Dense(64, activation='relu')(Prob_Dense)
# Prob_Dense = Dense(64, activation='sigmoid')(Prob_Dense)
# Prob_Dense = Dense(32, activation='relu')(Prob_Dense)
# Prob_Dense = Dense(96, activation='tanh')(Prob_Dense)#Ampliar para arquitectura encoder decoder
# Prob_Dense = Dense(16, activation='relu')(Prob_Dense)

# output = Dense(1, activation='linear')(Prob_Dense)

model = Model(
    inputs=Input0,
    # inputs=[DWT_Input,MFCCs_Input],
    outputs=output)

model.compile(optimizer=Adam(lr=.001),
              loss='mean_squared_error',
              metrics=['mape'])

# model.summary()
#%%

history = model.fit(x_train, 
                    y_train, 
                    validation_split=0.2, 
                    batch_size=64,
                    epochs =1000)


#%%
from matplotlib import pyplot as plt
figure()
#plot the training and validation accuracy and loss at each epoch
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

mse_neural, mape_neural = model.evaluate(x_test, y_test)
print('Mean squared error from neural net: ', mse_neural)
print('Mean absolute percentage error from neural net: ', mape_neural)
# acc = history.history['mae']
# val_acc = history.history['val_mae']
# plt.plot(epochs, acc, 'y', label='Training MAE')
# plt.plot(epochs, val_acc, 'r', label='Validation MAE')
# plt.title('Training and validation MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

#%%
############################################
#Predict on test data
predictions = model.predict(x_test).flatten()
figure()
plt.scatter(predictions,y_test)
lims=[0,100]
plt.plot(lims,lims)
plt.xlabel('predicted')
plt.ylabel('ture values')
plt.xlim(lims)
plt.ylim(lims)
# print("Predicted values are: ", predictions)
# print("Real values are: ", y_test[:10])
##############################################