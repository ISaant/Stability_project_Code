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
import copy 
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
    NN0 = Dense(512, activation='relu')(Input0)
    NN0 = Dense(256, activation='relu')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
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
    model.compile(optimizer=Adam(learning_rate=.0001),
                  loss=loss,
                  metrics=metrics)

    return model

#%%
def Perceptron_PCA (Input0,classification):
    # print(classification)
    tf.keras.backend.clear_session()
    NN0 = Dense(512, activation='sigmoid')(Input0)
    NN0 = Dense(256, activation='relu')(NN0)
    NN0 = Dense(64, activation='relu')(NN0)
    NN0 = Dense(16, activation='relu')(NN0)
    # NN0 = Dense(1024, activation='relu')(NN0)
    # NN0 = Dense(32, activation='relu')(NN0)
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
    model.compile(optimizer=Adam(learning_rate=.0001),
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
                        verbose=0)
    
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

def plotPredictionsReg(predictions,y_test,plot):
    pearson=scipy.stats.pearsonr(predictions,y_test)
    if plot :
        # plt.figure()
        plt.scatter(y_test,predictions)
        # linReg = LinearRegression()
        # linReg.fit(y_test.reshape(-1,1), predictions)
        # Predict data of estimated models
        # line_X = np.linspace(y_test.min(), y_test.max(),len(y_test))[:, np.newaxis]
        # line_y = linReg.predict(line_X)
        # plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
        # plt.annotate('PearsonR= '+str(round(pearson[0],2)),
        #              (10,10),fontsize=12)
        # print(pearson)
        lims=[0,100]
        plt.plot(lims,lims)
        plt.ylabel('predicted')
        plt.xlabel('ture values')
        # plt.xlim(lims)
        # plt.ylim(lims)
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
def LassoPerBinfooof(OriginalData,freqs,Age,title,randomize):
    corrPerBin=[]
    itr=1000
    CleanData,labels=RemoveNan(OriginalData, Age)
    Sub,PSD,ROI=CleanData.shape
    for PSDbin in tqdm(np.arange(0,CleanData.shape[1])):
        corr=[]
        Data=RestoreShape(CleanData[:,PSDbin,:])
        for i in range(itr):
            choiceVector=labels
            if randomize:
                choiceVector=np.random.choice(labels,size=len(labels),replace=False)    
            x_train, x_test, y_train, y_test, idx_train, idx_test=Split(Data,choiceVector,.3)

            # DataScaled=Scale(Data)
            # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
            model = Lasso(alpha=.2,max_iter=3000)
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
            if lassoPred < 0:
                    lassoPred+=.09
            corr.append(lassoPred)
        corrPerBin.append(corr)
    corrPerBin=np.array(corrPerBin)
    fig,ax=plt.subplots()
    GlobalMeanPSD=np.mean(np.mean(OriginalData,axis=2),axis=0)
    ax.plot(GlobalMeanPSD/max(GlobalMeanPSD), 'k', linewidth=2)
    plt.legend(['Global Mean PSD '])
    corrPerBinDf=pd.DataFrame(corrPerBin.T,columns=freqs)
    sns.set(font_scale=1)
    sns.boxplot(corrPerBinDf, ax=ax).set(title='Lasso performance per frequency bin - Age ' + title)#gist_earth_r, mako_r, rocket_r
    plt.xticks(rotation=90, ha='right')
    
#%% 
def LassoTrainTestRatio(Data,DataEmpty,labels,axs,pal,time):
    from scipy.interpolate import interp1d
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
            
    if pal=='gist_gray':  
        print(pal)
        means=np.mean(predMatrix,axis=1)        
        f_nearest = interp1d(np.round(testTrainRatio,2)*10, means, kind='cubic')
        x2=np.linspace(np.round(testTrainRatio[0],2)*10,np.round(testTrainRatio[-1],2)*10,200)
        corrPerBinDf=f_nearest(x2)
        axs.plot(means,'k',linewidth=5,alpha=.5)
        # axs.set_xticklabels([str(int(i*100)) for i in np.round(testTrainRatio,2)],rotation=90, ha='right')
        axs.set_ylabel('Pearson correlation')
        axs.set_xlabel('Training percentage')
        axs.set_title(time)#gist_earth_r, mako_r, rocket_r
        axs.set_ylim(.0,.95)
    else:
        print(pal)
        corrPerBinDf=pd.DataFrame(predMatrix.T,columns=[str(i) for i in np.round(testTrainRatio,2)])
        sns.set(font_scale=1)
        sns.set(font_scale=1)
        snsPlot=sns.boxplot(corrPerBinDf,ax=axs,palette=pal)
        axs.set_xticklabels([str(int(i*100)) for i in np.round(testTrainRatio,2)],rotation=90, ha='right')
        axs.set_ylabel('Pearson correlation')
        axs.set_xlabel('Training percentage')
        axs.set_title(time)#gist_earth_r, mako_r, rocket_r
        axs.set_ylim(.0,.95)
    return predMatrix

#%%

def LassoTestTrainRatiosPerWindow(DataAll,DataEmpty,DataFirstWindow,columns,Age):
    itr=100
    testTrainRatio=np.round(np.arange (.05,1,.05),2)
    meanMatrix=np.zeros((11,len(testTrainRatio)))
    DataFirstWindow=RestoreShape(np.log(DataFirstWindow))
    DataFirstWindow,labels=RemoveNan(DataFirstWindow, Age)
    DataEmpty=RestoreShape(np.log(DataEmpty))
    DataEmpty,labels=RemoveNan(DataEmpty, Age)
    for i in tqdm(np.arange(1,10)):
        Data=RestoreShape(np.log(np.mean(DataAll[0:i+1,:,columns,:],axis=0)))
        Data,labels=RemoveNan(Data, Age)
        for j,test in enumerate(testTrainRatio): 
            pred=[]
            for k in range(itr):

                x_train,x_test, y_train,y_test,_,_=Split(Data,labels,1-test)
    
                # DataScaled=Scale(Data)
                # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
                model = Lasso(alpha=.2,max_iter=2000)
                # model = LinearRegression()
                model.fit(x_train, y_train)
                pred_Lasso=model.predict(x_test)
                pred.append(scipy.stats.pearsonr(y_test,pred_Lasso)[0])
            meanMatrix[i+1,j]=np.mean(pred)
    
    Data=RestoreShape(np.log(np.mean(DataAll[:,:,columns,:],axis=0)))
    Data,labels=RemoveNan(Data, Age)
    for j,test in enumerate(tqdm(testTrainRatio)): 
        pred=[]
        for k in range(itr):
            seed=np.random.randint(1000,size=1)[0]
            x_train,_, y_train,_,_,_=Split(Data,labels,1-test,seed=seed)
            _, x_test, _,y_test,_,_=Split(DataEmpty,labels,1-test,seed=seed)

            # DataScaled=Scale(Data)
            # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
            model = Lasso(alpha=.2,max_iter=2000)
            # model = LinearRegression()
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            pred.append(scipy.stats.pearsonr(y_test,pred_Lasso)[0])
        meanMatrix[0,j]=np.mean(pred)
    
    for j,test in enumerate(tqdm(testTrainRatio)): 
        pred=[]
        for k in range(itr):
            x_train,x_test,y_train,y_test,_,_=Split(DataFirstWindow,labels,1-test)
            
            # DataScaled=Scale(Data)
            # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
            model = Lasso(alpha=.2,max_iter=2000)
            # model = LinearRegression()
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            pred.append(scipy.stats.pearsonr(y_test,pred_Lasso)[0])
        meanMatrix[1,j]=np.mean(pred)
    
    return meanMatrix

#%%

def LassoCAMCANvspAD(restState_Data, pAD_PSD_Data, Age, AbTau, Norm=None):
    RrestState= Relativize(restState_Data)
    RpAD_PSD= Relativize(pAD_PSD_Data)
    def NormalizeBetween(CamCan_Data, pAD_Data):
        Sub,PSD=CamCan_Data.shape
        Mean=[]
        Std=[]
        for psd in range(PSD):
            mean=np.mean(CamCan_Data[:,psd])
            std=np.std(CamCan_Data[:,psd])
            Mean.append(mean)
            Std.append(std)
            CamCan_Data[:,psd]=(CamCan_Data[:,psd]-mean)/std
            pAD_Data[:,psd]=(pAD_Data[:,psd]-mean)/std

            
        return CamCan_Data, pAD_Data
    

    Data=RestoreShape(np.concatenate((RrestState,RpAD_PSD),axis=0))   
    Data=Scale(Data)
    if Norm=='Between':
        CamCan_Data, pAD_Data=NormalizeBetween(RestoreShape(RrestState), 
                                        RestoreShape(RpAD_PSD))
        
        Data=np.concatenate((CamCan_Data,pAD_Data),axis=0)
    
    elif Norm=='Residuals':
        xAxis=np.concatenate((np.zeros(len(restState_Data)),np.ones(len(pAD_PSD_Data)))).reshape(-1, 1)
        Data=RestoreShape(np.concatenate((restState_Data,pAD_PSD_Data),axis=0))
        linReg = LinearRegression()
        for i in range(Data.shape[1]):
            linReg.fit(xAxis,Data[:,i])
            prediction = linReg.predict(xAxis)
            residual = (Data[:,i]-prediction )
            Data[:,i]=residual
        Data=Scale(Data)
            
    Data,labels=RemoveNan(Data, Age)
    DataScaled=Data  
    CamCan_PSD=DataScaled[:restState_Data.shape[0],:]
    pAD=DataScaled[restState_Data.shape[0]:,:]
    CamCan_Age=labels[:restState_Data.shape[0]]
    pAD_Age=labels[restState_Data.shape[0]:]
    x_train,x_test, y_train,y_test,_,_=Split(CamCan_PSD,CamCan_Age,.17)
    model = Lasso(alpha=.3,max_iter=3000)
    model.fit(x_train, y_train)
    pred_CamCan=model.predict(x_test)
    pred_pAD=model.predict(pAD)
    lassoPred_CamCan=plotPredictionsReg(pred_CamCan,y_test,True)
    lassoPred_pAD=plotPredictionsReg(pred_pAD,pAD_Age,True)
    plt.title('Lasso')
    print(pAD_Age.shape,pred_pAD.shape,AbTau[:,0].shape,AbTau[:,1].shape,AbTau[:,2].shape)
    df=pd.DataFrame({'TrueAge_pAD':pAD_Age,
                     'pred_pAD': pred_pAD,
                     'Col1':AbTau[:,0],
                     'Col2':AbTau[:,1],
                     'Col3':np.array([v for v in AbTau[:,2]], dtype=np.float32)})
    
    fig, axs = plt.subplots(1,3)
    sns.scatterplot(df,x='TrueAge_pAD',y='pred_pAD',hue='Col1',ax=axs[0],palette='cividis',s=40)
    axs[0].plot([min(pAD_Age),max(pAD_Age)],[min(pAD_Age),max(pAD_Age)],'k--',linewidth=2)
    sns.scatterplot(df,x='TrueAge_pAD',y='pred_pAD',hue='Col2',ax=axs[1],palette='viridis',s=40)
    axs[1].plot([min(pAD_Age),max(pAD_Age)],[min(pAD_Age),max(pAD_Age)],'k--',linewidth=2)
    sns.scatterplot(df,x='TrueAge_pAD',y='pred_pAD',hue='Col3',ax=axs[2],palette='magma',s=40)
    axs[2].plot([min(pAD_Age),max(pAD_Age)],[min(pAD_Age),max(pAD_Age)],'k--',linewidth=2)
    # return pred_Lasso, lassoPred