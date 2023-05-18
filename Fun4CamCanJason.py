#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:58:43 2023

@author: isaac
"""

import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from fooof import FOOOF
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model

#==============================================================================
def myReshape(array):
    [x,y]=array.shape
    newarray=np.zeros((606,300,68))
    for i,j in enumerate(np.arange(0,y,300)):
        newarray[:,:,i]=array[:,j:j+300]
        
    return newarray

#==============================================================================
def PltDist(demographics):
    
    
    sns.displot(data=demographics,x='age',kde=True)
    plt.title('Age Histogram')
    Age=demographics['age'].to_numpy()
    Catell=demographics['Catell_score']
    RoundAge=copy.copy(Age)
    RoundAge[RoundAge<30]=30
    for i in np.arange(30,90,10):
        print(i)
        RoundAge[np.logical_and(RoundAge>i, RoundAge<=i+10)]=(i+10)
    # RoundAge[RoundAge>80]=90
    demographics['Intervals']=RoundAge
    sns.displot(data=demographics,x='Catell_score', hue='Intervals',kind='kde', fill=True)
    plt.title('Catell Score Distribution')
    sns.displot(data=demographics,x='additional_acer', hue='Intervals',kind='kde', fill=True)
    plt.title('Acer Score Distribution')
    
    # plt.figure()
    sns.relplot(data=demographics,y='Catell_score', x='age', hue='Intervals')
    plt.title('Age-Catell Regression')
    idx=np.argwhere(np.isnan(Catell))
    Catell=np.delete(Catell, idx)
    Age=np.delete(Age, idx)
    rsq,pvalue=scipy.stats.pearsonr(Age,Catell)
    Age=Age.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Age, Catell)
    # Predict data of estimated models
    line_X = np.linspace(Age.min(), Age.max(),603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
                 (20,15),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                 (20,12),fontsize=12)
    plt.show()
#==============================================================================
def RemoveNan(Data,labels):
    idx=np.argwhere(np.isnan(labels))
    labels=np.delete(labels, idx)
    Data=np.delete(Data, idx,axis=0)
    return Data,labels
#==============================================================================
def fooof(Data, freqs, inBetween):
   Sub,PSD,ROI=Data.shape
   columnsInBetween= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1]+.5)) if x]
   #these new colums are for the creation of the matrix of all the fooof returns
   periodic= np.zeros((Sub,len(columnsInBetween),ROI))
   aperiodic=np.zeros((Sub,len(columnsInBetween),ROI))
   whitened=np.zeros((Sub,len(columnsInBetween),ROI))
   parameters=[]
   for roi in tqdm(range(ROI)):
       Roi=[]
       for sub in range(Sub):
           fm = FOOOF(max_n_peaks=6, aperiodic_mode='fixed',min_peak_height=0.15,verbose=False)
           fm.add_data(freqs, Data[sub,:,roi],inBetween) #freqs[0]<inBetween[:]<freqs[1]
           fm.fit(freqs, Data[sub,:,roi], inBetween)
           periodic[sub,:,roi]=fm._peak_fit
           aperiodic[sub,:,roi]=fm._ap_fit
           whitened[sub,:,roi]=fm.power_spectrum-fm._ap_fit
           exp = fm.get_params('aperiodic_params', 'exponent')
           offset = fm.get_params('aperiodic_params', 'offset')
           cfs = fm.get_params('peak_params', 'CF')
           pws = fm.get_params('peak_params', 'PW')
           bws = fm.get_params('peak_params', 'BW')
           Roi.append([exp,offset,cfs,pws,bws])
       parameters.append(Roi)
   freqsInBetween=fm.freqs
   return periodic, aperiodic, whitened, parameters, freqsInBetween

#==============================================================================

def psdPlot(freqs,Data):
    Sub,PSD,ROI=Data.shape
    # columnsInBetween= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1]+.5)) if x]
    figure()
    for roi in tqdm(range(ROI)):
        mean=np.mean(Data[:,:,roi],axis=0)
        plot(freqs,mean,alpha=.2)
        if roi == 0:
            Mean=mean
            continue
        Mean+=mean
    Mean/=(roi+1)
    plot(freqs,Mean,'k')
    

# ==============================================================================

def ACP (DataFrame,verbose,scatterMatrix,nPca):
    if verbose:
        
        print('----------------------')
        print('Media de cada variable')
        print('----------------------')
        print(DataFrame.mean(axis=0))
        
        
        print('-------------------------')
        print('Varianza de cada variable')
        print('-------------------------')
        print(DataFrame.var(axis=0))
    
    #Etiqueta de componentes a calcular
    # ==========================================================================
    Etiquetas=[]
    for i in range(len(DataFrame.keys())):
        Etiquetas.append('PC'+str(i+1))
    
    # Entrenamiento modelo PCA con escalado de los datos
    # ==============================================================================
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(DataFrame)
    
    # Se extrae el modelo entrenado del pipeline
    modelo_pca = pca_pipe.named_steps['pca']
    
    dfPca=pd.DataFrame(
        data    = modelo_pca.components_,
        columns = DataFrame.columns,
        index   = Etiquetas
    )
    if verbose:
        print(dfPca)
    
    # Heatmap componentes
    # ==============================================================================
    
    componentes = modelo_pca.components_
    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
        plt.imshow(componentes.T, cmap='viridis', aspect='auto')
        plt.yticks(range(len(DataFrame.columns)), DataFrame.columns)
        plt.xticks(range(len(DataFrame.columns)), np.arange(modelo_pca.n_components_) + 1)
        plt.grid(False)
        plt.colorbar();
    
    
    # Porcentaje de varianza explicada por cada componente
    # ==============================================================================
    if verbose:
        print('----------------------------------------------------')
        print('Porcentaje de varianza explicada por cada componente')
        print('----------------------------------------------------')
        print(modelo_pca.explained_variance_ratio_)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.bar(
            x      = np.arange(modelo_pca.n_components_) + 1,
            height = modelo_pca.explained_variance_ratio_
        )
    
        for x, y in zip(np.arange(len(DataFrame.columns)) + 1, modelo_pca.explained_variance_ratio_):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
    
        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_ylim(0, 1.1)
        ax.set_title('Porcentaje de varianza explicada por cada componente')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza explicada');
    
    # Porcentaje de varianza explicada acumulada
    # ==============================================================================
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    if verbose:
        print('------------------------------------------')
        print('Porcentaje de varianza explicada acumulada')
        print('------------------------------------------')
        print(prop_varianza_acum)
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.plot(
            np.arange(len(DataFrame.columns)) + 1,
            prop_varianza_acum,
            marker = 'o'
        )
    
        for x, y in zip(np.arange(len(DataFrame.columns)) + 1, prop_varianza_acum):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
            
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_title('Porcentaje de varianza explicada acumulada')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza acumulada');
    
    proyecciones = pca_pipe.transform(X=DataFrame)
    proyecciones = pd.DataFrame(
        proyecciones,
        columns = Etiquetas,
        index   = DataFrame.index
    )
    pro2use=proyecciones.iloc[:,:nPca]
    if verbose:
        print(proyecciones.head())
    
    if scatterMatrix:
        g=sns.pairplot(pro2use, hue="Cohort",corner=True)
        g.map_lower(sns.kdeplot, levels=6, color=".2")
    
    return proyecciones,pro2use,prop_varianza_acum

# =============================================================================
def myPCA (DataFrame,verbose,nPca):
    from sklearn import preprocessing
    scaled_data = preprocessing.scale(DataFrame)
    pca = PCA() # create a PCA object
    pca.fit(scaled_data) # do the math
    pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
     
    #########################
    #
    # Draw a scree plot and a PCA plot
    #
    #########################
     
    #The following code constructs the Scree plot
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    prop_varianza_acum = pca.explained_variance_ratio_.cumsum()
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    
    #the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(pca_data, columns=labels)
    pro2use=pca_df.iloc[:,:nPca]
    if verbose:
        
        plt.figure()
        plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.show()
        
        plt.figure()
        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title('My PCA Graph')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
         
        for sample in pca_df.index:
            plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.plot(
            np.arange(len(labels)) + 1,
            prop_varianza_acum,
            marker = 'o'
        )
    
        for x, y in zip(np.arange(len(labels)) + 1, prop_varianza_acum):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
            
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(pca.n_components_) + 1)
        ax.set_title('Percentage of cumulative explained variance')
        ax.set_xlabel('PCs')
        ax.set_ylabel('% of cumulative variance');
        plt.show()
        
    return pca_df, pro2use, prop_varianza_acum

 