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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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
            for a,alg in enumerate(Alg):
                Means=np.zeros((max(windows)))
                for j,win in enumerate(windows):
                    mean=np.mean(dframe.iloc[np.where((dframe['Num_of_Windows [30s]']==win) & (dframe['Algorithm']==alg))[0]]['Corr'])
                    Means[j]=mean
                    print(Means.shape)
                MeansDf=pd.DataFrame({'Corr':Means,'Windows': list(windows)})
                X=windows.reshape(-1, 1)
                regressor = SVR(kernel="poly", C=100, gamma="auto", degree=2, epsilon=0.1, coef0=1)
                regressor.fit(X,Means*100)
                yregPoly=regressor.predict(X)/100
                uu1=np.mean((yregPoly*100-Means*100)**2)

                regressor = SVR(kernel="linear")
                regressor.fit(X,Means*100)
                yregLin=regressor.predict(X)/100
                uu2=np.mean((yregLin*100-Means*100)**2)

                if uu1 < uu2 and yregPoly[0] < yregPoly[1] and np.diff(yregPoly)[0] > .01:
                    ax.plot(X, yregPoly,'k--',alpha=.5,label='Poly')# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
                    where=np.where(np.diff(yregPoly)<.01)[0][0]
                    plt.annotate(str(np.round(np.diff(yregPoly)[where],4)),
                                 (windows[where]+.5,Means[where]+.05),fontsize=12)
                    plt.vlines(windows[where]+.49, Means[where]+.05, Means[where]-.05,'r', alpha = .7)

                else:
                    ax.plot(X, yregLin,'k-',alpha=.5,label='Linear, Acc Steps =' +str(np.round(np.diff(yregLin)[0],3)))# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

                # if a == 0:
                ax.legend()

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