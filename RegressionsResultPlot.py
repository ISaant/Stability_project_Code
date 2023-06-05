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

def RegPlot (current_path):

    with open(current_path+'/Pickle/AgePredictions2.pickle', 'rb') as f:
        Age=pickle.load(f)
        
    with open(current_path+'/Pickle/CatellPredictions2.pickle', 'rb') as f:
        Catell=pickle.load(f)
        
    with open(current_path+'/Pickle/AcerPredictions2.pickle', 'rb') as f:
        Acer=pickle.load(f)
        
    # dfAge=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
    # dfCatell=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
    # dfAcer=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows'])
    #%%
    def mod_df(test): 
        Df=pd.DataFrame(columns=['Corr','Algorithm','Num_of_Windows [30s]'])
        for i in range(10):
            results=test[i]
            corr=np.reshape(results,(150,),order='F')
            iteration=np.ones(150)+i
            algorithm=np.reshape([['Lasso']*50,['Perceptron']*50,['RandomForest']*50],(150,)).astype(str)
            # df=pd.DataFrame({'Corr':corr,'Algorithm':algorithm,'Num_of_Windows [30s]':iteration})
            df=pd.DataFrame({'Corr':np.arctanh(corr),'Algorithm':algorithm,'Num_of_Windows [30s]':iteration})
            # lcls = locals()
            # exec('df'+target+'=pd.concat([df'+target+',df],ignore_index=True)',globals(),lcls)
            Df=pd.concat([Df,df],ignore_index=True)
        return Df
    dfAge=mod_df(Age)
    dfCatell=mod_df(Catell)
    dfAcer=mod_df(Acer)
    
    # # plt.close('all')
    def plotLine(dframe,ylim,order,title):
        
        def fit_line_poli(dframe, order, ax):
            windows = np.unique(dframe['Num_of_Windows [30s]']).astype(int)
            Alg,c=np.unique(dframe['Algorithm'],return_counts=True)
            algArray=[]
            for alg in Alg:
                Means=np.zeros((max(windows)))
                for j,win in enumerate(windows):
                    mean=np.mean(dframe.iloc[np.where((dframe['Num_of_Windows [30s]']==win) & (dframe['Algorithm']==alg))[0]]['Corr'])
                    Means[j]=mean
                MeansDf=pd.DataFrame({'Corr':Means,'Windows': list(windows)})
                sns.regplot(x="Windows", y="Corr",data=MeansDf, order=order, scatter=False, color = 'k', ci=False, line_kws=dict(alpha=1,linewidth=1,linestyle='--'),ax=ax)
            
                    
        fig, ax = plt.subplots()
        sns.lineplot(data=dframe,x='Num_of_Windows [30s]',y='Corr',hue='Algorithm', markers=True, palette="flare", ax=ax)
        fit_line_poli(dframe, order, ax)
        plt.ylabel('arctanh(Corr)')
        plt.ylim(ylim)
        plt.title(title)
    
    #%%
    plotLine(dfAge,[0.6,1.2],2,'Age regression')
    plotLine(dfCatell,[0.2,0.7],2, 'Catell Score regression')
    plotLine(dfAcer,[0,0.4],2,'Acer Score regression')

    #%%
    return dfAge
  