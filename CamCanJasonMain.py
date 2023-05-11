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
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from Fun4CamCanJason import *
from fooof import FOOOF
#%% Hyperparameters

#fooof 
inBetween=[2,50]

#PSD
freqs=np.arange(0,150,.5)
alfaBetaFreqs=[0,52]
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

demographics=pd.read_csv(path2Data+mainDir[0],header=None)
for e,file in enumerate(tqdm(restStateDir)):
    matrix=pd.read_csv(path2Data+mainDir[1]+'/'+file,header=None)
    if e == 0:
        print (e)
        restState=matrix
        continue
    restState+=matrix
restState=restState.to_numpy()
restState/=(e+1)
restState = myReshape(restState) #reshape into [ROI,subjects,PSD]
restStateCropped = restState[:,:,columns]

#%% Plot global mean and mean per ROI
figure(0)
ROI,Sub,PSD=restStateCropped.shape
for roi in tqdm(range(ROI)):
    mean=np.mean(restStateCropped[roi,:,:],axis=0)
    plot(freqs[columns],mean,alpha=.2)
    if roi == 0:
        Mean=mean
        continue
    Mean+=mean
Mean/=(roi+1)
plot(freqs[columns],Mean,'k')

#%%  
columnsInBetween= [i for i, x in enumerate((freqs[columns]>=inBetween[0]) & (freqs[columns]<inBetween[1])) if x]
newFreqs=freqs[columns]
newFreqs=newFreqs[columnsInBetween]
periodic= copy.copy(restStateCropped)
aperiodic=copy.copy(restStateCropped)
whitened=copy.copy(restStateCropped)
parameters=[]
for roi in tqdm(range(ROI)):
    Roi=[]
    for sub in range(Sub):
        fm = FOOOF(max_n_peaks=6, aperiodic_mode='fixed',min_peak_height=0.15)
        fm.add_data(freqs, restStateCropped[roi,sub,:],inBetween) #freqs[0]<inBetween[:]<freqs[1]
        fm.fit(freqs[columns], restStateCropped[roi,sub,:], inBetween)
        periodic[roi,sub,:]=fm._peak_fit
        aperiodic[roi,sub,:]=fm._ap_fit
        whitened[roi,sub,:]=fm.power_spectrum-fm._ap_fit
        exp = fm.get_params('aperiodic_params', 'exponent')
        offset = fm.get_params('aperiodic_params', 'offset')
        cfs = fm.get_params('peak_params', 'CF')
        pws = fm.get_params('peak_params', 'PW')
        bws = fm.get_params('peak_params', 'BW')
        Roi.append([exp,offset,cfs,pws,bws])
    parameters.append(Roi)
    
    
    
    
    
    
    
    
    