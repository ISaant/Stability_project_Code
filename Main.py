#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:19:49 2023

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from Fun2 import *
from FunClassifier import *


APer_Alpha_1peak = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=True, in_between=[1,50],
                      max_n_peaks=1, fit='fixed')

APer_Beta_1peak = GetStuff('beta', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=False, in_between=[1,50],
                      max_n_peaks=1, fit='fixed')


APer_Alpha_6peak = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=False, in_between=[1,250],
                      max_n_peaks=6, fit='knee')

APer_Beta_6peak = GetStuff('beta', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=False, in_between=[1,250],
                      max_n_peaks=6, fit='knee')

coeffs_Alpha=copy.copy(APer_Alpha_1peak.get_Parameters())
coeffs_Beta=copy.copy(APer_Beta_1peak.get_Parameters())
filter_banks_Alpha=APer_Alpha_6peak.filter_banks
filter_banks_Beta=APer_Beta_6peak.filter_banks
labels = APer_Alpha_6peak.periodic['Cohort'].to_numpy()
Par_Alpha=APer_Alpha_1peak.parameters
Par_Beta=APer_Beta_1peak.parameters
AlphaBetaCoeffDf=pd.concat([Par_Alpha.loc[:,'alphapws'],Par_Beta.loc[:,'betapws'],Par_Beta.loc[:,'Cohort']],axis=1)
sns.scatterplot(data=AlphaBetaCoeffDf,x='alphapws',y='betapws',hue='Cohort')
sns.displot(data=AlphaBetaCoeffDf,x='alphapws',y='betapws',hue='Cohort',kind='kde')

#%%
resultsCoeffAlpha=[]
resultsCoeffBeta=[]
resultsfbAlpha=[]
resultsfbBeta=[]
resultsCoeffAlphaBeta=[]
for i in range(10):
    resultsCoeffAlpha.append(nn_for_coefs(coeffs_Alpha))
    resultsCoeffBeta.append(nn_for_coefs(coeffs_Beta))
    resultsfbAlpha.append(nn_filterbank(filter_banks_Alpha,labels))
    resultsfbBeta.append(nn_filterbank(filter_banks_Beta,labels))
    resultsCoeffAlphaBeta.append(nn_for_PwsAlphaBeta(AlphaBetaCoeffDf))
