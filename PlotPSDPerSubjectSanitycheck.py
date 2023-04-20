#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:16:01 2023

@author: sflores
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path, os.pardir))
path = ParentPath+'/Stability-project_db/PSDfromTimeSeries_MultipleFeatures/'
effectSizePath=path+'0.8'+'_250_PSD/'
Dir=np.sort(os.listdir(effectSizePath))
freqs=np.linspace(0,250,(250*3)+1,endpoint=True)
Dataframe=pd.DataFrame()
in_between=[1,50]
columns= [i for i, x in enumerate((freqs>=in_between[0]) & (freqs<in_between[1])) if x]
freqs=freqs[columns]
allPSD=np.zeros((100,500,147))
for i,file in enumerate(Dir):
    main=pd.read_csv(effectSizePath+file,header=None)
    main=main[main.columns[columns]].to_numpy()
    allPSD[i,:,:]=main

for subject in range(allPSD.shape[1]):
    PSDs=[]
    plt.figure()
    for tw in range(allPSD.shape[0]):
        PSDs.append(allPSD[tw,subject,:])
    mean=np.mean(PSDs,axis=0)
    std=np.std(PSDs,axis=0)
    plt.plot(freqs,mean,'r')
    plt.fill_between(freqs,mean+std,mean-std,alpha=.5,color='r')
    
