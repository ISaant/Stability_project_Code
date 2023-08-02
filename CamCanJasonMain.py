#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:03:35 2023

@author: isaac
"""
import os
os.chdir('Documents/Doctorado_CIC/Internship/Sylvain/Stability-project/Stability_project_Code/')


import pandas as pd
import numpy as np
import copy
import seaborn as sns
import tensorflow as tf
import pickle
import math
from tensorflow import keras
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from Fun4CamCanJason import *
from FunClassifier4CamCanJason import *
from fooof import FOOOF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from TAR import TestAlgorithmsRegression
from RegressionsResultPlot import RegPlot

#%% Hyperparameters

#fooof 
inBetween=[1,40]

#PSD
freqs=np.arange(0,150,.5)
alfaBetaFreqs=[0,60]
columns= [i for i, x in enumerate((freqs>=alfaBetaFreqs[0]) & (freqs<alfaBetaFreqs[1])) if x]
columnsfooof= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1])) if x]
freqsCropped=freqs[columns]
freqsFooof=freqs[columnsfooof]
#columns is used to select the region of the PSD we are interested in

#%% Directories
current_path = os.getcwd()
parentPath=os.path.abspath(os.path.join(current_path,os.pardir))
path2Data=parentPath+'/Stability-project_db/CAMCAN_Jason_PrePro/'
path2Anatomical=parentPath+'/Stability-project_db/Anatomical_Features/'
AnatFile=os.listdir(path2Anatomical)
mainDir=np.sort(os.listdir(path2Data))
emptyRoomDir=np.sort(os.listdir(path2Data+mainDir[0]+'/'))
restStateDir=np.sort(os.listdir(path2Data+mainDir[2]+'/'))
taskDir=np.sort(os.listdir(path2Data+mainDir[3]+'/'))

#%% Read demografics 

demographics=pd.read_csv(path2Data+mainDir[1])
lineAge,ECatell=PltDistDemographics(demographics)
Catell=demographics['Catell_score'].to_numpy()
Age=demographics['age'].to_numpy()
Acer=demographics['additional_acer'].to_numpy()
Targets=['Catell','Age','Acer']

#%% Plot dist cortical thickness
CorticalThickness=pd.read_csv(path2Anatomical+AnatFile[0],header=None)
CorticalThickness_std=PltDistAnat(CorticalThickness,demographics) #Plot the map using R

#%% average Resting state using all time windows
for e,file in enumerate(tqdm(restStateDir)):
    matrix=myReshape(pd.read_csv(path2Data+mainDir[2]+'/'+file,header=None).to_numpy())[np.newaxis, :]
    if e == 0:
        # print (e)
        restStateAll=matrix 
        FirstWindow=matrix
        continue
    restStateAll=np.concatenate((restStateAll,matrix))
emptyRoom=pd.read_csv(path2Data+mainDir[0]+'/'+emptyRoomDir[0],header=None).to_numpy()
emptyRoom = myReshape(emptyRoom) #reshape into [Subjects,PSD,ROI]
emptyRoomCropped = emptyRoom[:,columns,:]
restStateOriginal=RestoreShape(np.mean(restStateAll,axis=0))
restState = myReshape(restStateOriginal) #reshape into [Subjects,PSD,ROI]
restStateNoOffset=np.empty(restState.shape)
Sub,PSD,ROI=restState.shape
for sub in range(Sub):
    for roi in range(ROI):
        # offset=np.mean(restState[sub,:,roi])
        offset=restState[sub,0,roi]
        restStateNoOffset[sub,:,roi]=restState[sub,:,roi]-offset
        # restStateNoOffset[sub,:,roi]+=abs(np.min(restStateNoOffset[sub,:,roi]))+.01
restStateCropped = restState[:,columns,:] # Select the band-width of interest
restStateCroppedNoOffset = restStateNoOffset[:,columns,:] # Select the band-width of interest
restStateFooof=restState[:,columnsfooof,:]
FirstWindow = np.squeeze(FirstWindow,axis=0)[:,columns,:]
#%% NNMF instead of PCA
nPca=100
# pca_df,pro2use,prop_varianza_acum=myPCA (np.log(restStateOriginal),True, nPca)
W = NNMatFac(restStateOriginal.to_numpy(),nPca)


#%%
#  FOOOF
    
periodic, aperiodic, whiten, parameters, freqsInBetween=fooof(restStateFooof, freqsFooof, inBetween)
mywhiten=myWhiten(restState[:,columns[2:],:],freqs[columns[2:]],parameters)
#%% Plot global mean PSD per Subject of just one ROI
psdAgeRangePlot_JustOneROI(freqs,restState[:,:,1],Age,'',False)
psdAgeRangePlot_JustOneROI(freqs,restStateNoOffset[:,:,1],Age,'w/o Offset',False)
psdAgeRangePlot_JustOneROI(freqs[columns[2:]],mywhiten[:,:,1],Age,'- Whiten',True)


#%% Plot global mean PSD per Subject
psdAgeRangePlot(freqs,restState,Age,'',False)
psdAgeRangePlot(freqs,restStateNoOffset,Age,'w/o Offset',False)
psdAgeRangePlot(freqs[columns[2:]],mywhiten,Age,' - Whiten',True)


#%% PSD mean per ROI after FOOOF
psdPlot(freqsInBetween, periodic)
psdPlot(freqsInBetween, aperiodic)
psdPlot(freqsInBetween, whiten)

#%% Plot mean rho based on the amount of time provided to the system as input
# Uncoment if you want to overwrite the picke files to plot the regresions
# TestAlgorithmsRegression(restStateDir,path2Data,mainDir,columns,)
# REMEMBER THAT THE exec() DOESNT WORK INSIDE FUNCTIONS! CHAGE THE CODE BEFORE USING IT AGAIN (not warking as a function)

#plot the results
from RegressionsResultPlot import RegPlot


df=RegPlot(current_path) #TODO Necesitas Hacer el goodness of fit
# MeanCorrMatrix(restStateCropped,current_path)
#%% Test Lasso with different "Data"
Data2Test=[restStateOriginal.to_numpy(),
           np.log(restStateOriginal.to_numpy()),
           restStateCropped,
           np.log(restStateCropped),
           periodic,
           aperiodic,
           whiten]
DataName=['Original','Log(Original)','LowPass[<50]','Log(LowPass[<50])','Rhythmic','Arhythmic','Whiten']
Test=[Age,Catell,Acer]
TheDf=pd.DataFrame(columns=['Corr','Data','Test'])
TestName=['Age','Catell','Acer']
for testName,test in zip(TestName,Test):
    for dataName,data2use in zip(tqdm(DataName),Data2Test):
        Data=RestoreShape(data2use)
        Data,labels=RemoveNan(Data, test)
        for i in range(100):
            x_train, x_test, y_train,y_test=Split(Data,labels,.2)
            # DataScaled=Scale(Data)
            # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
            model = Lasso(alpha=.2)
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
            TheDf.loc[len(TheDf)]=[lassoPred,dataName,testName]
        # plt.xlim([0,100])
        # plt.ylim([0,100])
        # plt.xlabel('True Score')
        # plt.ylabel('Predicted Score')
        # plt.annotate('r_sq='+str(round(LassoPred*100,4)), [20,80], fontsize=12)
        # plt.title('Lasso Regression on Age')
        
sns.set(font_scale=1)
sns.boxplot(TheDf,x='Test',y='Corr',hue='Data',palette="rocket_r").set(title='Lasso performance based on Dataset Processing')#gist_earth_r, mako_r, rocket_r
plt.xticks(rotation=15, ha='right')
#%% Search for the importance of the features

meanCorr=0
Data=RestoreShape(np.log(restStateCropped))
Data,labels=RemoveNan(Data, Age)
Data,_=RemoveNan(Data, Catell)
labels,Catell_noNan=RemoveNan(labels, Catell)
ExpectedCatell=np.zeros(len(labels))
for i,l in enumerate(labels):
    ExpectedCatell[i]=ECatell[np.where(lineAge==l)[0]]
itr=100
errorMatrix=np.empty((len(Data),itr))
errorMatrix.fill(np.nan)
predMatrix=np.empty((len(Data),itr))
predMatrix.fill(np.nan)
for i in tqdm(range(itr)):
    x_train, x_test, y_train,y_test,idx_train,idx_test=Split(Data,labels,.2)
    # DataScaled=Scale(Data)
    # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
    model = Lasso(alpha=.2)
    # model = LinearRegression()
    model.fit(x_train, y_train)
    pred_Lasso=model.predict(x_test)
    lassoPred=scipy.stats.pearsonr(y_test,pred_Lasso)[0]
    error = np.subtract(y_test,pred_Lasso)
    for idx,e,p in zip(idx_test,error,pred_Lasso):
        errorMatrix[idx,i]=e
        predMatrix[idx,i]=p
    meanCorr+=lassoPred
    # MAE.append(mae)
meanCorr/=itr
meanError=np.nanmean(errorMatrix,axis=1)
meanPred=np.round(np.nanmean(predMatrix,axis=1),2)

from Fun4CamCanJason import *
PlotErrorvsAge(meanPred,meanError,labels,Age,Catell_noNan,ExpectedCatell)

meanROICorr=0
MeanDiffAllbutOne=[]
for i in tqdm(range(68)):
    Data=np.delete(np.log(restStateCropped),i,axis=2)
    Data=RestoreShape(Data)
    Data,labels=RemoveNan(Data, Age)
    for _ in range(itr):
        x_train, x_test, y_train,y_test,_,_=Split(Data,labels,.2)
        model = Lasso(alpha=.2)
        model.fit(x_train, y_train)
        pred_Lasso=model.predict(x_test)
        lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
        meanROICorr+=lassoPred
    meanROICorr/=itr
    MeanDiffAllbutOne.append(meanCorr-meanROICorr)

MeanDiffJustOne=[]
AllROICorr=[]
for i in tqdm(range(68)):
    ROICorr=[]
    Data=np.log(restStateCropped[:,:,i])
    # Data=RestoreShape(Data)
    Data,labels=RemoveNan(Data, Age)
    for _ in range(itr):
        x_train, x_test, y_train,y_test,_,_=Split(Data,labels,.2)
        model = Lasso(alpha=.2)
        model.fit(x_train, y_train)
        pred_Lasso=model.predict(x_test)
        lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
        ROICorr.append(lassoPred)
    AllROICorr.append(ROICorr)
    meanROICorr=np.mean(ROICorr)
    MeanDiffJustOne.append(meanCorr-meanROICorr)

MeanDiffJustOne2=(1-np.array(MeanDiffJustOne))*meanCorr
MeanDiffAllbutOne2=1-np.array(MeanDiffAllbutOne) *meanCorr

MeanDiffJustOneER=[] #EmptyRoom
AllROICorrER=[]
for i in tqdm(range(68)):
    ROICorr=[]
    Data=np.log(restStateCropped[:,:,i])
    DataEmpty=np.log(emptyRoomCropped[:,:,i])
    # Data=RestoreShape(Data)
    Data,labels=RemoveNan(Data, Age)
    DataEmpty,labels=RemoveNan(DataEmpty, Age)
    for _ in range(itr):
        seed=np.random.randint(1000,size=1)[0]
        x_train,_, y_train,_,_,_=Split(Data,labels,.2,seed=seed)
        _, x_test, _,y_test,_,_=Split(DataEmpty,labels,.2,seed=seed)
        model = Lasso(alpha=.2)
        model.fit(x_train, y_train)
        pred_Lasso=model.predict(x_test)
        lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
        ROICorr.append(lassoPred)
    AllROICorrER.append(ROICorr)
    meanROICorr=np.mean(ROICorr)
    MeanDiffJustOneER.append(meanCorr-meanROICorr)

MeanDiffJustOneER2=(1-np.array(MeanDiffJustOneER))*meanCorr

#Correlation between empty room map and ROI map

CorrJO_Empty=[] #JustOne
for JO,ER in zip(AllROICorr,AllROICorrER):
    CorrJO_Empty.append(scipy.stats.pearsonr(JO,ER)[0])
CorrJO_Empty=np.array(CorrJO_Empty)
#%%
with open(current_path+'/Pickle/meanCorr_allFeatures.pickle', 'wb') as f:
    pickle.dump(meanCorr, f)
    
with open(current_path+'/Pickle/MeanDiff_AllbutOne.pickle', 'wb') as f:
    pickle.dump(MeanDiffAllbutOne, f)
    
with open(current_path+'/Pickle/MeanDiff_JustOne.pickle', 'wb') as f:
    pickle.dump(MeanDiffJustOne2, f)

#%% Train on all windows, test just on first window
FirstWindowCorr=0
DataTrain=RestoreShape(np.log(restStateCropped))
DataTrain,labels=RemoveNan(DataTrain, Age)
DataTest=RestoreShape(np.log(FirstWindow))
DataTest,labels=RemoveNan(DataTest, Age)
itr=100
for i in tqdm(range(itr)):
    x_train, _, y_train,_=Split(DataTrain,labels,.2,seed=i)
    _, x_test, _,y_test=Split(DataTest,labels,.2,seed=i)

    # DataScaled=Scale(Data)
    # x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)
    model = Lasso(alpha=.2)
    model.fit(x_train, y_train)
    pred_Lasso=model.predict(x_test)
    lassoPred=scipy.stats.pearsonr(pred_Lasso,y_test)[0]
    FirstWindowCorr+=lassoPred
FirstWindowCorr/=itr

#%% Training based on frequency bins

corrPerBin=[]
itr=50
CleanData,labels=RemoveNan(restStateCroppedNoOffset, Age)
Sub,PSD,ROI=CleanData.shape
for PSDbin in tqdm(np.arange(2,CleanData.shape[1])):
    corr=[]
    Data=RestoreShape(np.log(CleanData[:,PSDbin,:]))
    for i in range(itr):
        x_train, x_test, y_train, y_test, idx_train, idx_test=Split(Data,labels,.3)
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
GlobalMeanPSD=np.mean(np.mean(restStateCropped,axis=2),axis=0)
ax.plot(GlobalMeanPSD[2:]/max(GlobalMeanPSD[2:]), 'k', linewidth=2)
plt.legend(['Global Mean PSD'])
corrPerBinDf=pd.DataFrame(corrPerBin.T,columns=freqs[2:PSD])
sns.set(font_scale=1)
sns.boxplot(corrPerBinDf, ax=ax).set(title='Lasso performance per frequency bin - Age')#gist_earth_r, mako_r, rocket_r
plt.xticks(rotation=90, ha='right')

#%% Training based on frequency bins
LassoPerBin(restStateCropped,freqs,Age,'',randomize=False)
LassoPerBin(restStateCroppedNoOffset,freqs,Age,'w/o Offset',randomize=False)
LassoPerBinfooof(mywhiten,freqs[columns[2:]],Age,' Whiten',randomize=False)
LassoPerBin(restStateCropped,freqs,Age,'Random Permutations',randomize=True)

#%% Training based on frequency windows: 5 Hz w/ .5 Hz steps

corrPerBin=[]
itr=100
CleanData,labels=RemoveNan(restStateCropped, Age)
windowSize=int(5*(1/.5)) #5Hz windows * freq bin size in Hz/s = 10 datapoints,
step= int(.5*(1/.5)) #.5Hz step * freq bin size in Hz/s = 1 datapoint step
for PSDwinStart in tqdm(np.arange(0,CleanData.shape[1]-windowSize,step)):
    corr=[]
    Data=RestoreShape(np.log(CleanData[:,PSDwinStart:PSDwinStart+windowSize,:]))
    for i in range(itr):
        x_train, x_test, y_train, y_test, idx_train, idx_test=Split(Data,labels,.2)
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
GlobalMeanPSD=np.mean(np.mean(restStateCropped,axis=2),axis=0)
ax.plot(GlobalMeanPSD[int(windowSize/2):len(columns)-int(windowSize/2)]/max(GlobalMeanPSD[int(windowSize/2):len(columns)-int(windowSize/2)]), 'k', linewidth=2)
# ax.plot(GlobalMeanPSD[int(windowSize/2):100-int(windowSize/2)]/max(GlobalMeanPSD[2:]), 'k', linewidth=2)
plt.legend(['Global Mean PSD'])
corrPerBinDf=pd.DataFrame(corrPerBin.T,columns=freqs[int(windowSize/2):len(columns)-int(windowSize/2)])
sns.set(font_scale=1)
sns.boxplot(corrPerBinDf, ax=ax).set(title='Lasso performance per frequency window [ 5 Hz w/ .5 Hz steps]  - Age')#gist_earth_r, mako_r, rocket_r
plt.xticks(rotation=90, ha='right')


#%% Overfitting curve
fig,ax=plt.subplots(1,2,figsize=(17, 9))
DataAll=RestoreShape(np.log(restStateCropped))
DataAll,labels=RemoveNan(DataAll, Age)
DataFirstWindow=RestoreShape(np.log(FirstWindow))
DataFirstWindow,labels=RemoveNan(DataFirstWindow, Age)
DataEmptyRoom=RestoreShape(np.log(emptyRoomCropped))
DataEmptyRoom,labels=RemoveNan(DataEmptyRoom, Age)
predMatrixAllEmpty=LassoTrainTestRatio(DataAll,DataEmptyRoom,labels,ax[1],'gist_gray','300 sec')
predMatrixFirstWindowEmpty=LassoTrainTestRatio(DataFirstWindow,DataEmptyRoom,labels,ax[0],'gist_gray','30 sec')
predMatrixAll=LassoTrainTestRatio(DataAll,DataAll,labels,ax[1],'magma','300 sec')
predMatrixFirstWindow=LassoTrainTestRatio(DataFirstWindow,DataFirstWindow,labels,ax[0],'mako','30 sec')
# predMatrixFirstWindow=LassoTrainTestRatio(DataEmptyRoom,labels,ax,'gist_gray','30 sec')
plt.suptitle('Lasso performance under different Tests/Train ratios - Age')

#%% Training Percentage per Recorging Time
meanMatrix=LassoTestTrainRatiosPerWindow(restStateAll,emptyRoomCropped,FirstWindow,columns,Age)
colors=linear_gradient('#2100ff', '#ff0000', len(meanMatrix))
for i,mean in enumerate(meanMatrix):
    if i == 0:
        label='Empty room'
    else:
        label='Time = '+str(int(30*i))+" [s]"
    plt.plot(testTrainRatio,mean,color=colors['hex'][i],linewidth=5,alpha=.5,label=label)
plt.legend(fontsize=15)
plt.title('Lasso performance under different Tests/Train ratios and Recording times - Age',fontsize=17)
plt.xlabel('Train percentage',fontsize=15)
plt.ylabel('Pearson Correlation',fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

#%% Dictionary For ggseg


ROI={'bankssts_left':0,
'bankssts_right':1,
'caudalanteriorcingulate_left':0,
'caudalanteriorcingulate_right':1,
'caudalmiddlefrontal_left':0,
'caudalmiddlefrontal_right':1,
'cuneus_left':0,
'cuneus_right':1,
'entorhinal_left':0,
'entorhinal_right':1,
'frontalpole_left':0,
'frontalpole_right':1,
'fusiform_left':0,
'fusiform_right':1,
'inferiorparietal_left':0,
'inferiorparietal_right':1,
'inferiortemporal_left':0,
'inferiortemporal_right':1,
'insula_left':0,
'insula_right':1,
'isthmuscingulate_left':0,
'isthmuscingulate_right':1,
'lateraloccipital_left':0,
'lateraloccipital_right':1,
'lateralorbitofrontal_left':0,
'lateralorbitofrontal_right':1,
'lingual_left':0,
'lingual_right':1,
'medialorbitofrontal_left':0,
'medialorbitofrontal_right':1,
'middletemporal_left':0,
'middletemporal_right':1,
'paracentral_left':0,
'paracentral_right':1,
'parahippocampal_left':0,
'parahippocampal_right':1,
'parsopercularis_left':0,
'parsopercularis_right':1,
'parsorbitalis_left':0,
'parsorbitalis_right':1,
'parstriangularis_left':0,
'parstriangularis_right':1,
'pericalcarine_left':0,
'pericalcarine_right':1,
'postcentral_left':0,
'postcentral_right':1,
'posteriorcingulate_left':0,
'posteriorcingulate_right':1,
'precentral_left':0,
'precentral_right':1,
'precuneus_left':0,
'precuneus_right':1,
'rostralanteriorcingulate_left':0,
'rostralanteriorcingulate_right':1,
'rostralmiddlefrontal_left':0,
'rostralmiddlefrontal_right':1,
'superiorfrontal_left':0,
'superiorfrontal_right':1,
'superiorparietal_left':1,
'superiorparietal_right':1,
'superiortemporal_left':0,
'superiortemporal_right':1,
'supramarginal_left':0,
'supramarginal_right':1,
'temporalpole_left':0,
'temporalpole_right':1,
'transversetemporal_left':0,
'transversetemporal_right':1
}

keysList = list(ROI.keys())
for i,key in enumerate(ROI):
    ROI[key]=MeanDiffJustOne2[i]
import ggseg
ggseg.plot_dk(ROI, background='k', edgecolor='w', cmap='jet', 
              bordercolor='gray', ylabel='Age Predictibility (mm)', title='ROI Importance')
#%% Modify Dictionary based on Results

#%% Print Figure of the brain
#
#%% PCA statistics
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
