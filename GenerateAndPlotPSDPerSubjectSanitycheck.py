#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:59:58 2023

@author: sflores
"""

current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+"/Stability-project_db/simulatedTimeSeries_MultipleFeatures"
Path2savePSD=ParentPath+"/Stability-project_db/PSDfromTimeSeries_MultipleFeatures/0.8_250_PSD"
Dir=np.sort(os.listdir(Path2openSim))[1:]
fs=500
in_between=[1,50]
columns= [i for i, x in enumerate((freqs>=in_between[0]) & (freqs<in_between[1])) if x]
# freqs=freqs[columns]
# PSD=np.zeros((100,500,751))
PSDs=[]
for j,file in enumerate(tqdm(Dir)):
    Signal=pd.read_csv(Path2openSim+'/'+file,header=None).to_numpy()
    for i,signal in enumerate(Signal):
        freqs, psd_mean=welch(signal, fs,nperseg=1500,noverlap=750)
        PSDs.append(psd_mean)
    columns= [i for i, x in enumerate((freqs>=in_between[0]) & (freqs<in_between[1])) if x]
    mean=np.mean(PSDs,axis=0)[columns]
    std=np.std(PSDs,axis=0)[columns]
    plt.plot(freqs[columns],mean,'r')
    plt.fill_between(freqs[columns],mean+std,mean-std,alpha=.5,color='r')