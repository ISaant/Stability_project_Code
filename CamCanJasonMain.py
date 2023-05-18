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
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from Fun4CamCanJason import *
from FunClassifier4CamCanJason import *
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
PltDist(demographics)
Catell=demographics['Catell_score'].to_numpy()
Age=demographics['age'].to_numpy()
for e,file in enumerate(tqdm(restStateDir)):
    matrix=pd.read_csv(path2Data+mainDir[1]+'/'+file,header=None)
    if e == 0:
        print (e)
        restStateOriginal=matrix
        continue
    restStateOriginal+=matrix
restStateOriginal=restStateOriginal
restStateOriginal/=(e+1)
restState = myReshape(restStateOriginal.to_numpy()) #reshape into [Subjects,PSD,ROI]
restStateCropped = restState[:,columns,:] # Select the band-width of interest

#%% Plot global mean and mean per ROI
psdPlot(freqs[columns], restStateCropped)

#%%
nPca=68
pca_df,pro2use,prop_varianza_acum=myPCA (restStateOriginal,True, nPca)


#%% Delete nan from target drop same subject
target=Age
idx=np.argwhere(np.isnan(target))
labels=np.delete(target, idx)
Data=np.delete(pca_df, idx,axis=0)
DataScaled=Scale(Data)
#%%
from FunClassifier4CamCanJason import *
#%% Perceptron
x_train, x_test, y_train, y_test=Split(DataScaled[:,:70],labels,.3)
Input0=tf.keras.Input(shape=(x_train.shape[1],), )
model=Perceptron (Input0)
trainModel(model,x_train,y_train,300,True)
pred=evaluateModel(model,x_test,y_test)
plotPredictions(pred,y_test)
    
#%% CNN1D
DataCNN1D=restState[0]
DataCNN1D= Data.reshape(Data.shape[0], Data.shape[1], 1)
x_train, x_test, y_train, y_test=Split(DataCNN1D,labels,.3)
Input0=tf.keras.Input(shape=(x_train.shape[1],1),)
model=CCN1D(Input0)
trainModel(model,x_train,y_train,300,True)
pred=evaluateModel(model,x_test,y_test)
plotPredictions(pred,y_test)
#%% RandomForest
from sklearn.ensemble import RandomForestRegressor
x_train, x_test, y_train, y_test=Split(Data[:,:70],labels,.3,False)
model=RandomForestRegressor(n_estimators=100,random_state=30)
model.fit(x_train, y_train)
pred_Rf=model.predict(x_test)
plotPredictions(pred_Rf,y_test)
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
