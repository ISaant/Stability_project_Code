#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:21:07 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
from fooof import FOOOF
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import optimizers
# from tensorflow.keras import losses
# from tensorflow.keras import metrics
# from tensorflow.keras import models
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D
# from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
# from tensorflow.keras import Model
# from sklearn import metrics as Metrics
from git import Repo

path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(path, os.pardir))
PATH_OF_GIT_REPO = path+'/.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'comment from python script'

def git_push():
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        # repo.git.add(update=True)
        repo.git.add(all=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code')    

def git_pull():
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        origin = repo.remote(name='origin')
        origin.pull()
    except:
        print('Some error occured while pulling the code')  

class Stability_project:
   
    def __init__(self,band,initCase):
        self.path=ParentPath+'/Stability-project_db/simulated/'+band+'/'
        self.case=initCase


    def OpenCase(self):
        casePath=self.path+str(self.case)+'_250_PSD'
        Dir=np.sort(os.listdir(casePath))[2:-2]
        return Dir
    
    def APeriodic(self,freqs,DataFrame):
        dfPer=DataFrame.copy()
        dfAp=DataFrame.copy()
        for i in range(DataFrame.shape[0]):
            
            fm = FOOOF(max_n_peaks=1, aperiodic_mode='fixed')
            fm.add_data(freqs, np.array(DataFrame.iloc[i]),[1,40])
            fm.fit(freqs, np.array(DataFrame.iloc[i]), [1, 40])
            dfPer.iloc[i]=fm._peak_fit/5
            dfAp.iloc[i]=fm._ap_fit/5
        return dfPer,dfAp
    
    def Subjects_PerAper(self,timeWindows,idx,inBetween):
        print(self.case)
        DataFrameAp=pd.DataFrame()
        DataFramePer=pd.DataFrame()
        for win in tqdm(timeWindows):
            main=pd.read_csv(self.path+str(self.case)+'_250_PSD/'+win,header=None)
            freqs=a=np.linspace(0,250,(250*3)+1,endpoint=True)
            columns= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1])) if x]
            main=main[main.columns[columns]].iloc[np.concatenate((idx,idx+250))]
            dfPer,dfAp=self.APeriodic(freqs[(freqs>=inBetween[0]) & (freqs<inBetween[1])],main)
            DataFrameAp=pd.concat([DataFrameAp,dfAp],ignore_index=False)
            DataFramePer=pd.concat([DataFramePer,dfPer],ignore_index=False)

        
        DataFrameAp.reset_index(inplace=True)
        DataFrameAp.rename({'index':'id'},axis=1,inplace=True)
        DataFrameAp=DataFrameAp.groupby('id').mean()
        DataFrameAp['Cohort']=np.concatenate((np.zeros(len(idx)),np.ones(len(idx)))).astype(int)
        
        DataFramePer.reset_index(inplace=True)
        DataFramePer.rename({'index':'id'},axis=1,inplace=True)
        DataFramePer=DataFramePer.groupby('id').mean()
        DataFramePer['Cohort']=np.concatenate((np.zeros(len(idx)),np.ones(len(idx)))).astype(int)
        
        return DataFrameAp,DataFramePer,freqs[columns]
    
def ANN (DataFrame):
    
    PSD=np.array(DataFrame[DataFrame.columns[0:-2]])
    Labels=np.array(DataFrame[DataFrame.columns[-1]])
    X0_train, X0_test, train_labels, test_labels = train_test_split(PSD,
                                                                    Labels,
                                                                    test_size=.3)
    
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    
    MLP0_Input=tf.keras.Input(shape=(X0_train.shape[1],), name="fRateT")

    
    NN0 = Dense(512, activation='relu')(MLP0_Input)
    NN0 = Dense(128, activation='sigmoid')(NN0)
    NN0 = Dense(128, activation='relu')(NN0)
    NN0 = Dense(128, activation='tanh')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    NN0 = Dense(32, activation='relu')(NN0)
    Prob_Dense = Dense(32, activation='relu')(NN0)
    Prob_Dense = Dense(96, activation='tanh')(Prob_Dense)#Ampliar para arquitectura encoder decoder
    Prob_Dense = Dense(16, activation='relu')(Prob_Dense)
    output = Dense(2, activation='softmax',name='output')(Prob_Dense)
    
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
    
    