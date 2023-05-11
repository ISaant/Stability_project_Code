#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:58:43 2023

@author: isaac
"""

import numpy as np
import copy
from fooof import FOOOF

def myReshape(array):
    [x,y]=array.shape
    newarray=np.zeros((68,606,300))
    for i,j in enumerate(np.arange(0,y,300)):
        newarray[i,:,:]=array[:,j:j+300]
        
    return newarray

#
def fooof(Data, freqs, inBetween):
    
    periodic= copy.copy(Data)
    aperiodic=copy.copy(Data)
    whitened=copy.copy(Data)
    self.freqs=Data.freqs
    self.columns=Data.columns
    self.NFFT=Data.Dataframe_full.to_numpy()[:,:-1].shape[1]*2
    parameters=[]
    for i in tqdm(range(df.shape[0])):
        fm = FOOOF(max_n_peaks=self.max_n_peaks, aperiodic_mode=self.fit)
        fm.add_data(self.freqs, np.array(df.iloc[i]),inBetween)
        fm.fit(self.freqs, np.array(df.iloc[i]), inBetween)
        periodic.iloc[i]=fm._peak_fit
        aperiodic.iloc[i]=fm._ap_fit
        w=fm.power_spectrum-fm._ap_fit
        whitened.iloc[i]=w+abs(min(w))
        exp = fm.get_params('aperiodic_params', 'exponent')
        offset = fm.get_params('aperiodic_params', 'offset')
        cfs = fm.get_params('peak_params', 'CF')
        pws = fm.get_params('peak_params', 'PW')
        bws = fm.get_params('peak_params', 'BW')
        parameters.append([exp,offset,cfs,pws,bws])
        # parameters.append([cfs,pws,bws])
    periodic['Cohort']=cohort
    aperiodic['Cohort']=cohort
    parameters=np.array(parameters)
    parameters=np.concatenate((parameters,np.array([cohort]).T),axis=1)
    band=Generate_Data.band
    parameters=pd.DataFrame(parameters, columns=[band+'exp',band+'offset',band+'cfs',band+'pws',band+'bws','Cohort'])
    # parameters=pd.DataFrame(parameters, columns=[band+'cfs',band+'pws',band+'bws','Cohort'])
