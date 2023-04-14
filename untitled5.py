#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:21:39 2023

@author: isaac
"""

import numpy as np
from Fun2 import *
from FunClassifier import *
def addF1(results):
    # print('hello')

    def F1(array):
        f1=(2*array[2]*array[3])/(array[2]+array[3])
        return f1
    for i,rows in enumerate(results):
        results[i]=np.append(results[i],F1(results[i]))
    return results

# MeanF1=np.zeros((99,49))
# MeanAcc=np.zeros((99,49))
MeanF12=[]
MeanAcc2=[]
for i in np.arange(1,100,2):
    # for col,j in enumerate(np.arange(10,251,5)):
    APer_Alpha_1peak = GetStuff('alpha', Windows=int(i), 
                          sampleSize=250,seed=i, 
                          plot=False, in_between=[1,50],
                          max_n_peaks=1, fit='fixed')
    
    APer_Beta_1peak = GetStuff('beta', Windows=int(i), 
                          sampleSize=250,seed=i, 
                          plot=False, in_between=[1,50],
                          max_n_peaks=1, fit='fixed')

    Par_Alpha=APer_Alpha_1peak.parameters
    Par_Beta=APer_Beta_1peak.parameters
    AlphaBetaCoeffDf=pd.concat([Par_Alpha.loc[:,'alphapws'],Par_Beta.loc[:,'betapws'],Par_Beta.loc[:,'Cohort']],axis=1)
    #clean nans
    drop=[i for i,x in enumerate(AlphaBetaCoeffDf.isna().any(axis=1)) if x]
    AlphaBetaCoeffDf.drop(drop,inplace=True)
    resultsCoeffAlphaBeta=[]
    resultsSVM=[]
    for k in range(5):
        resultsCoeffAlphaBeta.append(nn_for_PwsAlphaBeta(AlphaBetaCoeffDf))
        # resultsSVM.append(SVM(AlphaBetaCoeffDf))
    resultsCoeffAlphaBeta=np.array(addF1(list(resultsCoeffAlphaBeta)))
    # MeanF1[i-1,col]=np.mean(resultsCoeffAlphaBeta[:,-1])
    # MeanAcc[i-1,col]=np.mean(resultsSVM)
    MeanF12.append(resultsCoeffAlphaBeta[:,-1])
    # MeanAcc2.append(np.mean(resultsSVM))
       