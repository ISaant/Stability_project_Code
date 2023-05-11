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
import mne
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+"/Stability-project_db/simulatedTimeSeries_MultipleFeatures"
Path2savePSD=ParentPath+"/Stability-project_db/PSD_Multitapers_fromTimeSeries_MultipleFeatures/0.8_250_PSD"
Dir=np.sort(os.listdir(Path2openSim))[1:]
fs=500
PSD=np.zeros((100,500,235))
for j,file in enumerate(tqdm(Dir)):
    Signal=pd.read_csv(Path2openSim+'/'+file,header=None).to_numpy()
    for i,sig in enumerate(Signal):
        psd, freqs=mne.time_frequency.psd_array_multitaper(sig, sfreq=fs, fmin=1, fmax=40, adaptive=True, n_jobs=-1, verbose=False )
        PSD[i,j,:]=psd

for k in tqdm(range(PSD.shape[0])):
    if k < 10:
        np.savetxt(Path2savePSD+'/PSD_0' + str(k) + '_simulation_6sec.csv', PSD[k,:,:], delimiter=",")
    else:
        np.savetxt(Path2savePSD+'/PSD_' + str(k) + '_simulation_6sec.csv', PSD[k,:,:], delimiter=",")
        
np.savetxt(Path2savePSD+'/frequencyArray_6sec.csv', freqs, delimiter=",")