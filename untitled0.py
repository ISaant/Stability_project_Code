#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:00:53 2023

@author: sflores
"""
import pandas as pd
with open(parentPath+'/Pickle/AgePredictions.pickle', 'rb') as f:
    Age=pickle.load(f)
    
with open(parentPath+'/Pickle/CatellPredictions.pickle', 'rb') as f:
    Catell=pickle.load(f)
    
with open(parentPath+'/Pickle/AcerPredictions.pickle', 'rb') as f:
    Acer=pickle.load(f)
    
dfAge=pd.DataFrame(columns=['Corr','Algorithm','Iteration'])
dfCatell=pd.DataFrame(columns=['Corr','Algorithm','Iteration'])
dfAcer=pd.DataFrame(columns=['Corr','Algorithm','Iteration'])
Targets=['Age','Catell','Acer']
#%%
for target in Targets:
    test=eval(target)    
    for i in range(10):
        results=test[i]
        corr=np.reshape(results,(150,),order='F')
        iteration=np.ones(150)+i
        algorithm=np.reshape([['Lasso']*50,['Perceptron']*50,['RandomForest']*50],(150,)).astype(str)
        df=pd.DataFrame({'Corr':corr,'Algorithm':algorithm,'Num_of_Windows [30s]':iteration})
        exec('df'+target+'=pd.concat([df'+target+',df],ignore_index=True)')

#%%
# plt.close('all')
def plotLine(df,ylim,title):
    plt.figure()
    sns.lineplot(data=df,x='Num_of_Windows [30s]',y='Corr',hue='Algorithm', markers=True, palette="flare")
    plt.ylim(ylim)
    plt.title(title)

plotLine(dfAge,[0,.85],'Age regression')
plotLine(dfCatell,[0,.85], 'Catell Score regression')
plotLine(dfAcer,[0,.85], 'Acer Score regression')