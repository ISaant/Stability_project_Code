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
import pickle
from tensorflow import keras
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from Fun4CamCanJason import *
from FunClassifier4CamCanJason import *
from fooof import FOOOF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
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

#%% Read demografics 

demographics=pd.read_csv(path2Data+mainDir[0])
PltDist(demographics)
Catell=demographics['Catell_score'].to_numpy()
Age=demographics['age'].to_numpy()
Acer=demographics['additional_acer'].to_numpy()
Targets=['Catell','Age','Acer']
#%% average Resting state using all time windows
ACER=[]
AGE=[]
CATELL=[]
# Input0=tf.keras.Input(shape=(70,), )
# modelNN=Perceptron (Input0,False)
for i in tqdm(range(10)):
    TimeWindows=restStateDir[0:i+1]
    if i == 0:
        print(i)
        TimeWindows=[restStateDir[0]]
    for e,file in enumerate(TimeWindows):
        matrix=pd.read_csv(path2Data+mainDir[1]+'/'+file,header=None)
        if e == 0:
            # print (e)
            restStateOriginal=matrix
            continue
        restStateOriginal+=matrix
    restStateOriginal=restStateOriginal
    restStateOriginal/=(e+1)
    restState = myReshape(restStateOriginal.to_numpy()) #reshape into [Subjects,PSD,ROI]
    restStateCropped = restState[:,columns,:] # Select the band-width of interest

# Plot global mean and mean per ROI
# psdPlot(freqs[columns], restStateCropped)


    nPca=100
    # pca_df,pro2use,prop_varianza_acum=myPCA (np.log(restStateOriginal),True, nPca)
    W = NNMatFac(restStateOriginal.to_numpy(),nPca)
    
# Delete nan from target drop same subject, we will use all regions =========
    for target in Targets:
        label=eval(target)
        exec(target+'Reg=[]')
        print (target)
        # Data,labels=RemoveNan(np.log(restStateOriginal), label)
        # DataScaled=Scale(Data)
        # Data3,labels2=RemoveNan(pca_df, label)
        # DataScaled3=Scale(Data3)
        Data,labels=RemoveNan(W, label)
        DataScaled=Scale(Data)
        for j in range(50):
        # Data,labels=RemoveNan(np.log(restStateOriginal), label)
            #x_train, x_test, y_train,y_test =Split(DataScaled,labels,.2)
            #x_train2, x_test2, y_train2,y_test2=Split(Data2[:,:50],labels2,.2)
            x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)

        # Lasso
            model = Lasso(alpha=.2)
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            LassoPred=plotPredictionsReg(pred_Lasso,y_test)
    
    
        # Perceptron Regression
            Input0=tf.keras.Input(shape=(x_train.shape[1],), )
            modelNN=Perceptron (Input0,False)
            trainModel(modelNN,x_train,y_train,500,True)
            # predNN=evaluateRegModel(model,x_test,y_test)
            predNN = modelNN.predict(x_test).flatten()
            NNPred=plotPredictionsReg(predNN,y_test)
            
        # Random Forest
            model=RandomForestRegressor(n_estimators=20)
            model.fit(x_train, y_train)
            pred_Rf=model.predict(x_test)
            RFPred=plotPredictionsReg(pred_Rf,y_test)
            plt.close('all')
            exec(target+'Reg.append([LassoPred, NNPred, RFPred ])')
            
        A=np.array(eval(target+'Reg'))
        # exec(target.upper()+'.append([[np.mean(A[:,0]),np.std(A[:,0])], [np.mean(A[:,1]),np.std(A[:,1])] ,[np.mean(A[:,2]),np.std(A[:,2])]])')
        exec(target.upper()+'.append(A)')
    with open(parentPath+'/Pickle/AgePredictions.pickle', 'wb') as f:
        pickle.dump(AGE, f)
    with open(parentPath+'/Pickle/CatellPredictions.pickle', 'wb') as f:
        pickle.dump(CATELL, f)
    with open(parentPath+'/Pickle/AcerPredictions.pickle', 'wb') as f:
        pickle.dump(ACER, f)
    # Perceptron Classification
    
    # from tensorflow.keras.utils import to_categorical
        # labelsClass=demographics['Intervals'].to_numpy()
        # labelsClass=((labelsClass-min(labelsClass))/10).astype(int)
        # x_train, x_test, y_train, y_test=Split(DataScaled[:,:70],labels,.3)
        # y_train=to_categorical(y_train,num_classes=7)
        # y_test=to_categorical(y_test,num_classes=7)
        # Input0=tf.keras.Input(shape=(x_train.shape[1],), )
        # model=Perceptron (Input0,True)
        # trainModel(model,x_train,y_train,300,True)
        # pred=evaluateClassModel(model,x_test,y_test)
# plotPredictions(pred,y_test)
    
# CNN1D
# DataCNN1D=restState[0]
# DataCNN1D= Data.reshape(Data.shape[0], Data.shape[1], 1)
# x_train, x_test, y_train, y_test=Split(DataCNN1D,labels,.3)
# Input0=tf.keras.Input(shape=(x_train.shape[1],1),)
# model=CCN1D(Input0)
# trainModel(model,x_train,y_train,300,True)
# pred=evaluateRegModel(model,x_test,y_test)
# plotPredictionsReg(pred,y_test)
# RandomForest






#%%
#  FOOOF

periodic, aperiodic, whitened, parameters, freqsInBetween=fooof(restStateCropped, freqs[columns], inBetween)


# Plot global mean and mean per ROI after FOOOF
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

# myPCA statistics
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
