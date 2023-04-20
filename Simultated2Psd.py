#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:12:41 2023

@author: sflores
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm import tqdm

current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+"/Stability-project_db/simulatedTimeSeries_MultipleFeatures"
Path2savePSD=ParentPath+"/Stability-project_db/PSDfromTimeSeries_MultipleFeatures/0.8_250_PSD"
Dir=np.sort(os.listdir(Path2openSim))[1:]
fs=500
PSD=np.zeros((100,500,751))
for j,file in enumerate(tqdm(Dir)):
    Signal=pd.read_csv(Path2openSim+'/'+file,header=None).to_numpy()
    for i,signal in enumerate(Signal):
        freq_mean, psd_mean=welch(signal, fs,nperseg=1500,noverlap=750)
        PSD[i,j,:]=psd_mean

for k in tqdm(range(PSD.shape[0])):
    if k < 10:
        np.savetxt(Path2savePSD+'/PSD_0' + str(k) + '_simulation_6sec.csv', PSD[k,:,:], delimiter=",")
    else:
        np.savetxt(Path2savePSD+'/PSD_' + str(k) + '_simulation_6sec.csv', PSD[k,:,:], delimiter=",")