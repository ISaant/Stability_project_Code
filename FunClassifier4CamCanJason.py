#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:26:03 2023

@author: isaac
"""

import tensorflow as tf
import scipy 
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
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
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D,Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import Model
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
from Fun4CamCanJason import *
from tqdm import tqdm

#%% Scale data
def Scale(Data):
    
    scaler=StandardScaler()
    scaler.fit(Data)
    Data=scaler.transform(Data)
    return Data

#%% Split data
def Split(Data,labels,testSize,seed=None):
    idx = np.arange(len(Data))
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(Data, labels, idx, test_size=testSize,random_state=seed)
    return  x_train, x_test, y_train, y_test, idx_train, idx_test
   
#%% Perceptron
def Perceptron (Input0,classification):
    # print(classification)
    tf.keras.backend.clear_session()
    NN0 = Dense(2048, activation='linear')(Input0)
    NN0 = Dense(256, activation='sigmoid')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    output = Dense(1, activation='linear')(NN0)
    loss='mean_squared_error',
    metrics=['mape']
    if classification:
        output = Dense(7, activation='softmax')(NN0)
        loss='categorical_crossentropy',
        metrics=[Precision(),Recall()]
    model = Model(
        inputs=Input0,
        outputs=output)
    
    
    # print(model.summary())
    model.compile(optimizer=Adam(learning_rate=.001),
                  loss=loss,
                  metrics=metrics)

    return model

#%% CNN1D
def CCN1D (Input0):
    
    
   
    CNN0 = Conv1D(128, 3, activation="relu",padding="same")(Input0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = Conv1D(64, 3, activation="relu",padding="same")(CNN0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = MaxPooling1D(pool_size=2)(CNN0)
    CNN0 = Conv1D(32, 3, activation="relu",padding="same")(CNN0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = Conv1D(16, 3, activation="relu",padding="same")(CNN0)
    CNN0 = BatchNormalization() (CNN0)
    CNN0 = Dropout(0.1)(CNN0)
    CNN0 = MaxPooling1D(pool_size=2)(CNN0)
    Flat = Flatten()(CNN0)
    
    cnn = Model(Input0, Flat)
    cnn.summary()

    NN0 = Dense(512, activation='relu')(Flat)
    NN0 = Dense(128, activation='sigmoid')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    # NN0 = Dense(32, activation='relu')(NN0)
    NN1 = Dense(16, activation='relu')(NN0)
    output = Dense(1, activation='linear')(NN0)

    model = Model(inputs=Input0,
                outputs=output)
    
    model.compile(optimizer=Adam(lr=.0001),
                  loss='mean_squared_error',
                  metrics=['mape'])

    return model

#%%
def trainModel(model,x_train,y_train,epochs,plot):
    keras.backend.clear_session()
    history = model.fit(x_train, 
                        y_train, 
                        validation_split=0.2, 
                        batch_size=64,
                        epochs =epochs,
                        verbose=1)
    
    if plot:
        
        plt.figure()
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
    
def evaluateRegModel(model,x_test,y_test):
    mse_neural, mape_neural = model.evaluate(x_test, y_test, verbose=0)
    # print('Mean squared error from neural net: ', mse_neural)
    # print('Mean absolute percentage error from neural net: ', mape_neural)
    predictions = model.predict(x_test).flatten()
    return predictions

def evaluateClassModel(model,x_test,y_test):
    # print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    print(results)

#%% Function to plot predictions

def plotPredictionsReg(predictions,y_test):
    plt.figure()
    plt.scatter(predictions,y_test)
    pearson=scipy.stats.pearsonr(predictions,y_test)
    # print(pearson)
    lims=[0,100]
    plt.plot(lims,lims)
    plt.xlabel('predicted')
    plt.ylabel('ture values')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.show()
    return pearson[0]

#%%
def LassoPerBin(OriginalData,freqs,Age,title,randomize):
    corrPerBin=[]
    itr=50
    CleanData,labels=RemoveNan(OriginalData, Age)
    Sub,PSD,ROI=CleanData.shape
    for PSDbin in tqdm(np.arange(2,CleanData.shape[1])):
        corr=[]
        Data=RestoreShape(np.log(CleanData[:,PSDbin,:]))
        for i in range(itr):
            choiceVector=labels
            if randomize:
                choiceVector=np.random.choice(labels,size=len(labels),replace=False)    
            x_train, x_test, y_train, y_test, idx_train, idx_test=Split(Data,choiceVector,.3)

            # DataScaled=Scale(Data)
            # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
            model = Lasso(alpha=.2)
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
            corr.append(lassoPred)
        corrPerBin.append(corr)
    corrPerBin=np.array(corrPerBin)
    fig,ax=plt.subplots()
    GlobalMeanPSD=np.mean(np.mean(OriginalData,axis=2),axis=0)
    ax.plot(GlobalMeanPSD[2:]/max(GlobalMeanPSD[2:]), 'k', linewidth=2)
    plt.legend(['Global Mean PSD '])
    corrPerBinDf=pd.DataFrame(corrPerBin.T,columns=freqs[2:PSD])
    sns.set(font_scale=1)
    sns.boxplot(corrPerBinDf, ax=ax).set(title='Lasso performance per frequency bin - Age ' + title)#gist_earth_r, mako_r, rocket_r
    plt.xticks(rotation=90, ha='right')
    
#%% 
def LassoTrainTestRatio(Data,DataEmpty,labels,axs,pal,time):
    itr=100
    testTrainRatio=np.arange (.05,1,.05)
    predMatrix=np.zeros((len(testTrainRatio),itr))
    for i,test in enumerate(tqdm(testTrainRatio)): 
        for j in range(itr):
            seed=np.random.randint(1000,size=1)[0]
            x_train,_, y_train,_,_,_=Split(Data,labels,1-test,seed=seed)
            _, x_test, _,y_test,_,_=Split(DataEmpty,labels,1-test,seed=seed)

            # DataScaled=Scale(Data)
            # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
            model = Lasso(alpha=.2,max_iter=2000)
            # model = LinearRegression()
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            predMatrix[i,j]=scipy.stats.pearsonr(y_test,pred_Lasso)[0]
    corrPerBinDf=pd.DataFrame(predMatrix.T,columns=[str(i) for i in np.round(testTrainRatio,2)])
    sns.set(font_scale=1)

    
    if str(type(axs))=="<class 'numpy.ndarray'>":
        for ax in axs:
            snsPlot=sns.boxplot(corrPerBinDf,ax=ax,palette=pal)
            for patch in snsPlot.artists:
                fc = patch.get_facecolor()
                patch.set_facecolor(mpl.colors.to_rgba(fc, 0.7))
            
            ax.set_xticklabels([str(int(i*100)) for i in np.round(testTrainRatio,2)],rotation=90, ha='right')
            ax.set_ylabel('Pearson correlation')
            ax.set_xlabel('Training percentage')
            ax.set_title(time)#gist_earth_r, mako_r, rocket_r
            ax.set_ylim(.0,.95)
    else:
        sns.set(font_scale=1)
        snsPlot=sns.boxplot(corrPerBinDf,ax=axs,palette=pal)
        for patch in snsPlot.artists:
            fc = patch.get_facecolor()
            patch.set_facecolor(mpl.colors.to_rgba(fc, 0.7))
        
        axs.set_xticklabels([str(int(i*100)) for i in np.round(testTrainRatio,2)],rotation=90, ha='right')
        axs.set_ylabel('Pearson correlation')
        axs.set_xlabel('Training percentage')
        axs.set_title(time)#gist_earth_r, mako_r, rocket_r
        axs.set_ylim(.0,.95)
    return predMatrix