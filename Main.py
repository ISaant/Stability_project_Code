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

#%%
APer_Alpha_1peak = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=False, in_between=[1,50],
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
SVM(AlphaBetaCoeffDf)

#%%
resultsCoeffAlpha=[]
for i in range(10):
    resultsCoeffAlpha.append(nn_for_coefs(coeffs_Alpha))
    
#%%
resultsCoeffBeta=[]
for i in range(10):
    resultsCoeffBeta.append(nn_for_coefs(coeffs_Beta))
    
#%%
resultsfbAlpha=[]
for i in range(10):
    resultsfbAlpha.append(nn_filterbank(filter_banks_Alpha,labels))
#%%
resultsfbBeta=[]
for i in range(10):
    resultsfbBeta.append(nn_filterbank(filter_banks_Beta,labels))
#%%
resultsCoeffAlphaBeta=[]
for i in range(10):
    resultsCoeffAlphaBeta.append(nn_for_PwsAlphaBeta(AlphaBetaCoeffDf))
    
#%%
def addF1(results):
    print('hello')

    def F1(array):
        f1=(2*array[2]*array[3])/(array[2]+array[3])
        return f1
    for i,rows in enumerate(results):
        results[i]=np.append(results[i],F1(results[i]))
    return results

resultsCoeffAlpha= np.array(addF1(list(resultsCoeffAlpha)))
resultsCoeffBeta=np.array(addF1(list(resultsCoeffBeta)))
resultsfbAlpha=np.array(addF1(list(resultsfbAlpha)))
resultsfbBeta=np.array(addF1(list(resultsfbBeta)))
resultsCoeffAlphaBeta=np.array(addF1(list(resultsCoeffAlphaBeta)))

#%%
plt.figure(figsize=(15,8))
columns=['f1Alpha_FooofCoeffs', 'f1Beta_FooofCoeffs', 'f1Alpha_Mffcs','f1Beta_Mffcs','f1Alph+Beta_Pwr']
F1scores=pd.DataFrame(np.array([resultsCoeffAlpha[:,-1],resultsCoeffBeta[:,-1],resultsfbAlpha[:,-1],resultsfbBeta[:,-1],resultsCoeffAlphaBeta[:,-1]]).T,
                      columns=columns)
boxplot=sns.boxplot(F1scores,palette="flare")
boxplot.axes.set_title("F1 Score BoxPlot, Efect Size = 0.8, N=all, Time=all",fontsize=20)
boxplot.set_xlabel("features",fontsize=15)
boxplot.set_ylabel("F1 Score",fontsize=15)
boxplot.tick_params(labelsize=12)