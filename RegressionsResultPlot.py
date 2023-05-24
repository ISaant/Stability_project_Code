#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:00:53 2023

@author: sflores
"""
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def RegPlot (parentPath):

    with open(parentPath+'/Pickle/AgePredictions2.pickle', 'rb') as f:
        Age=pickle.load(f)
        
    with open(parentPath+'/Pickle/CatellPredictions2.pickle', 'rb') as f:
        Catell=pickle.load(f)
        
    with open(parentPath+'/Pickle/AcerPredictions2.pickle', 'rb') as f:
        Acer=pickle.load(f)
        
    # dfAge=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
    # dfCatell=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
    # dfAcer=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
    
    def mod_df(test): 
        Df=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
        for i in range(10):
            results=test[i]
            corr=np.reshape(results,(150,),order='F')
            iteration=np.ones(150)+i
            algorithm=np.reshape([['Lasso']*50,['Perceptron']*50,['RandomForest']*50],(150,)).astype(str)
            df=pd.DataFrame({'Corr':corr,'Algorithm':algorithm,'Num_of_Windows [30s]':iteration})
            # lcls = locals()
            # exec('df'+target+'=pd.concat([df'+target+',df],ignore_index=True)',globals(),lcls)
            Df=pd.concat([Df,df],ignore_index=True)
        return Df
    dfAge=mod_df(Age)
    dfCatell=mod_df(Catell)
    dfAcer=mod_df(Acer)
    
    # # plt.close('all')
    def plotLine(dframe,ylim,title):
        plt.figure()
        sns.lineplot(data=dframe,x='Num_of_Windows [30s]',y='Corr',hue='Algorithm', markers=True, palette="flare")
        plt.ylim(ylim)
        plt.title(title)
    
    
    plotLine(dfAge,[0,.85],'Age regression')
    plotLine(dfCatell,[0,.85], 'Catell Score regression')
    plotLine(dfAcer,[0,.85], 'Acer Score regression')

    return dfAge
  