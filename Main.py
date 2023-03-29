#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:19:49 2023

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from Fun import *

bands= ['alpha','beta']

for band in bands:
    Cases=[0.2, 0.5, 0.8] #iterar esto al final
    Dir=Stability_project(band,0.8).OpenCase()
    # Dir = OpenCase(case=Cases[0]) 
    timeWindows=np.random.choice(Dir,size=50,replace=False) #iterar esto primero
    idx=np.sort(np.random.choice(np.arange(0,250),size=250,replace=False))
    DataFrame=Stability_project(band,0.8).Subjects(timeWindows,idx)
    
    plt.figure()
    PSD=np.array(DataFrame[DataFrame.columns[0:-2]])
    freqs=np.arange(1,40,.3)
    meanA=np.mean(PSD[:250],axis=0)
    meanB=np.mean(PSD[250:],axis=0)
    stdA=np.std(PSD[:250],axis=0)
    stdB=np.std(PSD[250:],axis=0)
    plt.plot(freqs,meanA,'r')
    plt.plot(freqs,meanB,'g')
    plt.fill_between(freqs,meanA+stdA,meanA-stdA,alpha=.5,color='r')
    plt.fill_between(freqs,meanB+stdB,meanB-stdB,alpha=.5,color='g')