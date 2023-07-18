#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:04:56 2023

@author: isaac
"""
def plotPredMatrix(predMatrix,testTrainRatio,plot,pal,axs):
    from scipy.interpolate import interp1d
    if plot=='line':  
        print(pal)
        means=np.mean(predMatrix,axis=1)        
        C=19
        f_nearest = interp1d(np.round(testTrainRatio,2)*C, means, kind='cubic')
        x2=np.linspace(np.round(testTrainRatio[0],2),np.round(testTrainRatio[-1],2)*C,200)
        corrPerBinDf=f_nearest(x2)
        corrPerBinDf=pd.DataFrame({'means':means,
                                   'ratios':testTrainRatio})
        sns.lineplot(corrPerBinDf,y='means',x='ratios',ax=axs,palette=pal,linewidth=5)
        axs.plot(means)
        # axs.set_xticklabels([str(int(i*100)) for i in np.round(testTrainRatio,2)],rotation=90, ha='right')
        axs.set_ylabel('Pearson correlation')
        axs.set_xlabel('Training percentage')
        # axs.set_title(time)#gist_earth_r, mako_r, rocket_r
        # axs.set_ylim(.0,.95)
    else:
        print(pal)
        corrPerBinDf=pd.DataFrame({'means':np.ravel(predMatrix),
                                   'ratios':np.repeat([i for i in np.round(testTrainRatio,2)],4)})
        sns.set(font_scale=1)
        snsPlot=sns.boxplot(corrPerBinDf,x='ratios',y='means',ax=axs,palette=pal)
        # axs.set_xticklabels([str(int(i*100)) for i in np.round(testTrainRatio,2)],rotation=90, ha='right')
        axs.set_ylabel('Pearson correlation')
        axs.set_xlabel('Training percentage')
        # axs.set_title(time)#gist_earth_r, mako_r, rocket_r
        axs.set_ylim(.0,.95)

fig,ax=plt.subplots(1,2,figsize=(17, 9))
testTrainRatio=np.arange (.05,1,.05)
plotPredMatrix(predMatrixAll,testTrainRatio,'boxplot','magma',ax[1])
plotPredMatrix(predMatrixFirstWindow,testTrainRatio,'boxplot','mako',ax[0])
plotPredMatrix(predMatrixAllEmpty,testTrainRatio,'line','gist_gray',ax[1])
plotPredMatrix(predMatrixFirstWindowRmpty,testTrainRatio,'line','gist_gray',ax[0])