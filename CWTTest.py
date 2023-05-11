#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:18:29 2023

@author: sflores
"""
import os
import pywt as pw
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tqdm
import pickle
from numpy import pi,cos,sin,random
from scipy import signal

def mirrowing(signal):
    initial=list(signal)
    mirrowed = list(signal[::-1])
    initial.extend(mirrowed)
    mirrowed.extend(initial)
    return np.array(mirrowed)

def freq2scale(motherWavelet,fs,fmax,fmin):
    cf=pw.central_frequency(motherWavelet)
    Scale=[]
    freqs=[fmax,fmin]
    for fr in freqs:
        # Scale.append(np.round(np.log((cf*fs)/fr))/np.log(2))
        Scale.append(np.log((cf*fs)/fr)/np.log(2))
    return Scale

def CWT(sig,motherWavelet,fs,fmax,fmin):

    widths = []
    Scales=freq2scale(motherWavelet, fs, fmax, fmin)
    v=np.arange(Scales[0],Scales[1],1)
    M=32
    for J in v:   #generate the scales
            a1=[]
            for m in np.arange(1,M): 
                a1.append(2**(J+(m/M)))
            
            widths.append(a1)
    widths=np.array(widths)
    widths=widths.reshape(widths.shape[1]*widths.shape[0],)
    # widths=np.arange(40,80,1/8)# widths = []
    sp=1/fs
    # Cwtmatr=[]
    # print(i)
    # sig= signal.hilbert(signal.filtfilt(b60,a60,sig))
    sig= signal.hilbert(sig)
    cwtmatr, freqs = pw.cwt(sig, widths, motherWavelet, sampling_period=sp)
    cwtmatr_real=abs(cwtmatr)
    for rows in range(cwtmatr.shape[0]):
        r=np.copy(cwtmatr[rows])
        cwtmatr[rows]=r/np.mean(r)
    # var=np.var(cwtmatr,axis=1)
    #cwtmatr=abs(cwtmatr)
    #cwtmatr=((cwtmatr/mn[:, np.newaxis])+np.min(cwtmatr))/np.max(cwtmatr)
    # cwtmatr=(cwtmatr+np.min(cwtmatr))/np.max(cwtmatr)
    # cwtmatr=cwtmatr/np.max(cwtmatr)
    # X.append(sig)
    # Cwtmatr.append(cwtmatr)
            
    
    return cwtmatr,freqs
    
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+"/Stability-project_db/simulatedTimeSeries_MultipleFeatures"
Path2saveCWT=ParentPath+"/Stability-project_db/CWTfromTimeSeries_MultipleFeatures/0.8_250_PSD"
Dir=np.sort(os.listdir(Path2openSim))[1:]
fs=500
# PSD=np.zeros((100,500,1501))
for j,file in enumerate(tqdm(Dir)):
    Signal=pd.read_csv(Path2openSim+'/'+file,header=None).to_numpy()
    for i,sig in enumerate(Signal):
        x=mirrowing(sig)
        Cwtmatr,freqs=CWT(x,'morl',fs,40,4)
        cwtmatr=np.abs(Cwtmatr[:,len(sig):len(sig)*2])

# for k in tqdm(range(PSD.shape[0])):
#     if k < 10:
#         np.savetxt(Path2savePSD+'/PSD_0' + str(k) + '_simulation_6sec.csv', PSD[k,:,:], delimiter=",")
#     else:
#         np.savetxt(Path2savePSD+'/PSD_' + str(k) + '_simulation_6sec.csv', PSD[k,:,:], delimiter=",")
        
# np.savetxt(Path2savePSD+'/PSD_' + str(k) + '_frequencyArray_6sec.csv', freq_mean, delimiter=",")


t=np.arange(0,6,1/fs)
# addNoise = 0.025*random.rand(len(t))
sig  = np.concatenate((np.cos(2 * np.pi * 8 * t[0:1500]),np.sin(2 * np.pi * 20 * t[1500:]))) #+ np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
x=mirrowing(sig)
# x = x+addNoise;
Cwtmatr,freqs=CWT(x,'morl',fs,40,4)
cwtmatr=np.abs(Cwtmatr[:,len(sig):len(sig)*2])

# fig,ax=plt.subplots(2,1,figsize=(8,8))
# ax[0].plot(x)
# ax[1].imshow(cwtmatr,  cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max()) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
y = freqs
X, Y = np.meshgrid(t, y)
Z = cwtmatr

ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0.1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()








