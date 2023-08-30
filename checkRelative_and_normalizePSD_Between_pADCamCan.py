#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:37:04 2023

@author: sflores
"""

restState_Data=np.log(Relativize(restStateCropped))
pAD_PSD_Data=np.log(Relativize(pAD_PSD[:,columns,:]))
xAxis=np.concatenate((np.zeros(len(restState_Data)),np.ones(len(pAD_PSD_Data)))).reshape(-1, 1)
Data=RestoreShape(np.concatenate((restState_Data,pAD_PSD_Data),axis=0))
for i in range(Data.shape[1]):
    linReg = LinearRegression()
    linReg.fit(xAxis,Data[:,i])
    prediction = linReg.predict(xAxis)
    residual = Data[:,i]-prediction 
    Data[:,i]=residual+np.mean(Data[:len(restState_Data),i])
# Data=Scale(Data)Relativize
psdPlot(freqs[4:80],myReshape(Data,704)[:len(restState_Data),4:80,:])
psdPlot(freqs[4:80],myReshape(Data,704)[len(restState_Data):,4:80,:])