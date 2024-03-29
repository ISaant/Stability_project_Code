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
import statistics
from fooof import FOOOF
from tqdm import tqdm
from matplotlib.pyplot import plot, figure
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model

# Gradientes de color =======================================================
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex, n):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)
#==============================================================================
def myReshape(array,num_subjects=606):
    [x,y]=array.shape
    newarray=np.zeros((num_subjects,int(y/68),68))
    for i,j in enumerate(np.arange(0,y,int(y/68))):
        newarray[:,:,i]=array[:,j:j+int(y/68)]
        
    return newarray

#==============================================================================

def reformat_AD(Matrix):
    ROIs,PSD=Matrix.shape
    reformatedMatrix=np.zeros((int(ROIs/68),int(PSD*68)))
    for subj in range(int(ROIs/68)):
        subjPSD=[]
        for roi in np.linspace(subj*68,subj*68+68,68,endpoint=False,dtype=int):
            subjPSD.extend(Matrix[roi,:])
        reformatedMatrix[subj]=np.array(subjPSD)
    return reformatedMatrix

#==============================================================================

def Relativize(Data):
    Sub,PSD,ROI=Data.shape
    RelData=copy.copy(Data)
    for sub in range(Sub):
        for roi in range(ROI):
            RelData[sub,:,roi]/=sum(RelData[sub,:,roi])
    return RelData

#==============================================================================

def RestoreShape(Data):
    if len(Data.shape)>2:
        [x,y,z]=Data.shape
        newarray=np.zeros((x,y*z))
        for i,j in enumerate(np.arange(0,y*z,y)):
            newarray[:,j:j+y]=Data[:,:,i]
        return newarray
    else:
        return Data

#==============================================================================
def PltDistDemographics(demographics):
    
    sns.displot(data=demographics,x='age',kde=True)
    plt.title('Age Histogram')
    Age=demographics['age'].to_numpy()
    Acer=demographics['additional_acer'].to_numpy()
    Catell=demographics['Catell_score'].to_numpy()
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
    Acer=np.delete(Acer, idx)
    idx=np.argwhere(np.isnan(Acer))
    Catell=np.delete(Catell, idx)
    Age=np.delete(Age, idx)
    Acer=np.delete(Acer, idx)
    rsq,pvalue=scipy.stats.pearsonr(Age,Catell)
    Age=Age.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Age, Catell)
    # Predict data of estimated models
    line_age = line_X = np.round(np.arange(Age.min()-5, Age.max()+5,.01),2)[:, np.newaxis]
    line_predCatell = linReg.predict(line_X)
    plt.plot(line_age,line_predCatell,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
                 (20,15),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                 (20,12),fontsize=12)

    sns.relplot(data=demographics,x='Catell_score', y='additional_acer', hue='Intervals')
    plt.title('Catell-Acer Regression')
    rsq,pvalue=scipy.stats.pearsonr(Catell,Acer)
    Catell=Catell.reshape(-1,1)
    # Acer=Acer.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Catell,Acer)
    # Predict data of estimated models
    line_X = np.linspace(Catell.min(), Catell.max(),603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
                 (20,17),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                 (20,10),fontsize=12)

    sns.relplot(data=demographics,x='age', y='additional_acer', hue='Intervals')
    plt.title('Age-Acer Regression')
    Age=Age.reshape(Age.shape[0],)
    rsq,pvalue=scipy.stats.pearsonr(Age,Acer)
    Age=Age.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Age,Acer)
    # Predict data of estimated models
    line_X = np.linspace(Age.min(), Age.max(),603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
                 (20,77),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                 (20,70),fontsize=12)
    plt.ylim([60,110])
    plt.show()
    return line_age, line_predCatell

#==============================================================================

def PltDistAnat(CorticalThickness,demographics):
    ROI_Names=list(pd.read_csv('ROI_Names.csv',header=None)[0])
    Catell=demographics['Catell_score'].to_numpy()
    Age=demographics['age'].to_numpy()
    MeanCorThick=np.mean(CorticalThickness,axis=1)
    idxOutliers=np.array([119,491])
    MeanCorThick=np.delete(MeanCorThick,idxOutliers)
    Catell=np.delete(Catell, idxOutliers)
    Age=np.delete(Age, idxOutliers)
    MeanCorticalThickness=pd.DataFrame({'MeanCorThick':MeanCorThick,
                               'Age':Age,
                               'Catell': Catell})
    RoundAge=copy.copy(Age)
    RoundAge[RoundAge<30]=30
    for i in np.arange(30,90,10):
        print(i)
        RoundAge[np.logical_and(RoundAge>i, RoundAge<=i+10)]=(i+10)
    # RoundAge[RoundAge>80]=90
    MeanCorticalThickness['AgeIntervals']=RoundAge
    sns.displot(data=MeanCorticalThickness,x='MeanCorThick', hue='AgeIntervals',kind='kde', fill=True,palette='mako')
    plt.title('Mean Cortical Thickness Score Distribution')


    # plt.figure()
    sns.relplot(data=MeanCorticalThickness,y='MeanCorThick', x='Age', hue='AgeIntervals',palette='mako')
    plt.title('Age-Cortical Thickness Regression')
    
    idx=np.argwhere(np.isnan(Catell))
    Catell=np.delete(Catell, idx)
    Age=np.delete(Age, idx)
    MeanCorThick=np.delete(MeanCorThick,idx)
    rsq,pvalue=scipy.stats.pearsonr(Age,MeanCorThick)
    Age=Age.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Age,MeanCorThick)
    # Predict data of estimated models
    line_age  = np.round(np.arange(Age.min()-5, Age.max()+5,.01),2)[:, np.newaxis]
    line_predCorThick = linReg.predict(line_age)
    plt.plot(line_age,line_predCorThick,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR= '+str(round(rsq,2)),
                  (30,2.3),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                  (30,2.2),fontsize=12)
    
    sns.relplot(data=MeanCorticalThickness,y='MeanCorThick', x='Catell', hue='AgeIntervals',palette='mako')
    plt.title('Age-Cortical Thickness Regression')
    rsq,pvalue=scipy.stats.pearsonr(Catell,MeanCorThick)
    Catell=Catell.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Catell,MeanCorThick)
    # Predict data of estimated models
    line_Catell = np.round(np.arange(Catell.min()-2, Catell.max()+2,.01),2)[:, np.newaxis]
    line_predCorThick = linReg.predict(line_Catell)
    plt.plot(line_Catell,line_predCorThick,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR= '+str(round(rsq,2)),
                  (30,2.3),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                  (30,2.2),fontsize=12)

    CorticalThickness_std=pd.DataFrame({'CorThick_std':[statistics.stdev(x)for x in np.transpose(CorticalThickness.to_numpy())],
                    'ROI_Names': ROI_Names})
    return CorticalThickness_std

#==============================================================================
def RemoveNan(Data,labels):
    idx=np.argwhere(np.isnan(labels))
    labels=np.delete(labels, idx)
    Data=np.delete(Data, idx,axis=0)
    return Data,labels

# def RandomizeFeaturesInROI(Data):
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

def myWhiten(DataOriginal,freqs,parameters):
    Data=copy.copy(DataOriginal)
    for roi in range(len(parameters)):
        for sub in range(len(parameters[0])):
            exp=parameters[roi][sub][0]
            offset=parameters[roi][sub][1]
            arythmic=offset-np.log10(freqs**exp)
            Data[sub,:,roi]=10**(np.log10(Data[sub,:,roi])-arythmic)
            
    return Data

#==============================================================================

def psdPlot(freqs,Data):
    Sub,PSD,ROI=Data.shape
    # columnsInBetween= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1]+.5)) if x]
    figure()
    for sub in tqdm(range(Sub)):
        mean=np.mean(Data[sub,:,:],axis=1)
        # plot(np.log(freqs),np.log(mean),alpha=.2)
        # plot(np.log(freqs),np.log(mean))
        plot(freqs,mean,alpha=.2)

        # plt.show()
        # plt.pause(1)

        if sub == 0:
            Mean=mean
            continue
        Mean+=mean
    Mean/=(Sub)
    # plot(np.log(freqs),np.log(Mean),'k')
    # plot(np.log(freqs),np.log(Mean),'k')
    plot(freqs,Mean,'k')
# 

    plt.title('Global PSD')
    plt.xlabel('log(Frequencies [Hz])')
    plt.ylabel('log(Power)')
    
    

# =============================================================================

def psdAgeRangePlot(freqs,Data,Age,plusTitle,loglog,AgeStep=2.5):

    Sub,PSD,ROI=Data.shape
    fig = figure(figsize=(10,7))
    plt.suptitle('PSD per Age Group '+plusTitle)
    Data=Data[Age.argsort(),:,:]
    AgeSorted=np.sort(Age)
    AgeStep=2.5
    SubGroup=np.arange(min(Age)+(AgeStep*2),max(Age)-AgeStep*2,AgeStep)
    colors=linear_gradient('#ffd89b', '#19547b', np.floor(len(SubGroup)-1).astype(int)+1)
    grid = plt.GridSpec(2, 12, wspace=0.4, hspace=0.3)
    freqRange1=np.where((freqs>=5)*(freqs<=35))[0]
    freqRange2=np.where((freqs>49)*(freqs<51))[0]
    for i,s in enumerate(tqdm(SubGroup)):
        # s=int(np.round(s))
        # print(i)
        AgeRange=np.where((AgeSorted>=s)*(AgeSorted<s+AgeStep))[0]
        mean=np.mean(np.mean(Data[AgeRange,:,:],axis=2),axis=0)
        # meanAge=np.round(np.mean(Age[s:s+SubGroup]))
        plt.subplot(grid[0, :])
        if loglog:
            plot(np.log(freqs[:PSD]),np.log(mean),color=colors['hex'][i], label = str(s+AgeStep))
            plt.subplot(grid[1, 3:10])
            plot(np.log(freqs[freqRange1]),np.log(mean[freqRange1]),color=colors['hex'][i])
            plt.subplot(grid[1, 10:])
            plot(np.log(freqs[freqRange2]),np.log(mean[freqRange2]),color=colors['hex'][i])
            plt.yticks([])
            # plot(freqs[:PSD],mean,color=colors['hex'][i],alpha=.7, label = str(s+AgeStep))
        else:
            plot(freqs[:PSD],mean,color=colors['hex'][i], label = str(s+AgeStep))
            plt.subplot(grid[1, 3:10])
            plot(freqs[freqRange1],mean[freqRange1],color=colors['hex'][i])
            plt.subplot(grid[1, 10:])
            plot(freqs[freqRange2],mean[freqRange2],color=colors['hex'][i])
            plt.yticks([])
        if i == 0:
            Mean=mean
            continue
        Mean+=mean
    Mean/=(i+1)
    plt.subplot(grid[0, :])
    if loglog:
        plot(np.log(freqs[:PSD]),np.log(Mean),'k',label='mean')
        # plt.title('freqs [0:150]')
        plt.xlabel('log(Frequencies  [0:150] Hz)')
        plt.ylabel('log(Power)')
        plt.legend()
        plt.subplot(grid[1, 3:10])
        plot(np.log(freqs[freqRange1]),np.log(Mean[freqRange1]),'k',)
        plt.xlabel('log(Frequencies [5:35]Hz)')
        plt.ylabel('log(Power)')
        plt.subplot(grid[1, 10:])
        plot(np.log(freqs[freqRange2]),np.log(Mean[freqRange2]),'k',)
        plt.xlabel('log(Freq [49:51]Hz)')
    else:
        plot(freqs[:PSD],Mean,'k',label='mean')
        # plt.title('freqs [0:150]')
        plt.xlabel('Frequencies  [0:150] Hz')
        plt.ylabel('Power')
        plt.legend()
        plt.subplot(grid[1, 3:10])
        plot(freqs[freqRange1],Mean[freqRange1],'k',)
        plt.xlabel('Frequencies [5:35]Hz')
        plt.ylabel('Power')
        plt.subplot(grid[1, 10:])
        plot(freqs[freqRange2],Mean[freqRange2],'k',)
        plt.xlabel('Freq [49:51]Hz')
    # plot(freqs[:PSD],Mean,'k',label='mean')
    # plt.subplot(grid[:, 3])
    
# =============================================================================
def psdAgeRangePlot_JustOneROI(freqs,Data,Age,plusTitle,loglog,AgeStep=2.5):

    Sub,PSD=Data.shape
    fig = figure(figsize=(10,7))
    plt.suptitle('PSD per Age Group '+plusTitle)
    Data=Data[Age.argsort(),:]
    AgeSorted=np.sort(Age)
    AgeStep=2.5
    SubGroup=np.arange(min(Age)+(AgeStep*2),max(Age)-AgeStep*2,AgeStep)
    colors=linear_gradient('#ffd89b', '#19547b', np.floor(len(SubGroup)-1).astype(int)+1)
    grid = plt.GridSpec(2, 12, wspace=0.4, hspace=0.3)
    freqRange1=np.where((freqs>=5)*(freqs<=35))[0]
    freqRange2=np.where((freqs>49)*(freqs<51))[0]
    for i,s in enumerate(tqdm(SubGroup)):
        # s=int(np.round(s))
        # print(i)
        AgeRange=np.where((AgeSorted>=s)*(AgeSorted<s+AgeStep))[0]
        mean=np.mean(Data[AgeRange,:],axis=0)
        # meanAge=np.round(np.mean(Age[s:s+SubGroup]))
        plt.subplot(grid[0, :])
        if loglog:
            plot(np.log(freqs[:PSD]),np.log(mean),color=colors['hex'][i], label = str(s+AgeStep))
            plt.subplot(grid[1, 3:10])
            plot(np.log(freqs[freqRange1]),np.log(mean[freqRange1]),color=colors['hex'][i])
            plt.subplot(grid[1, 10:])
            plot(np.log(freqs[freqRange2]),np.log(mean[freqRange2]),color=colors['hex'][i])
            plt.yticks([])
            # plot(freqs[:PSD],mean,color=colors['hex'][i],alpha=.7, label = str(s+AgeStep))
        else:
            plot(freqs[:PSD],mean,color=colors['hex'][i], label = str(s+AgeStep))
            plt.subplot(grid[1, 3:10])
            plot(freqs[freqRange1],mean[freqRange1],color=colors['hex'][i])
            plt.subplot(grid[1, 10:])
            plot(freqs[freqRange2],mean[freqRange2],color=colors['hex'][i])
            plt.yticks([])
        if i == 0:
            Mean=mean
            continue
        Mean+=mean
    Mean/=(i+1)
    plt.subplot(grid[0, :])
    if loglog:
        plot(np.log(freqs[:PSD]),np.log(Mean),'k',label='mean')
        # plt.title('freqs [0:150]')
        plt.xlabel('log(Frequencies  [0:150] Hz)')
        plt.ylabel('log(Power)')
        plt.legend()
        plt.subplot(grid[1, 3:10])
        plot(np.log(freqs[freqRange1]),np.log(Mean[freqRange1]),'k',)
        plt.xlabel('log(Frequencies [5:35]Hz)')
        plt.ylabel('log(Power)')
        plt.subplot(grid[1, 10:])
        plot(np.log(freqs[freqRange2]),np.log(Mean[freqRange2]),'k',)
        plt.xlabel('log(Freq [49:51]Hz)')
    else:
        plot(freqs[:PSD],Mean,'k',label='mean')
        # plt.title('freqs [0:150]')
        plt.xlabel('Frequencies  [0:150] Hz')
        plt.ylabel('Power')
        plt.legend()
        plt.subplot(grid[1, 3:10])
        plot(freqs[freqRange1],Mean[freqRange1],'k',)
        plt.xlabel('Frequencies [5:35]Hz')
        plt.ylabel('Power')
        plt.subplot(grid[1, 10:])
        plot(freqs[freqRange2],Mean[freqRange2],'k',)
        plt.xlabel('Freq [49:51]Hz')
    # plot(freqs[:PSD],Mean,'k',label='mean')
    # plt.subplot(grid[:, 3])

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

#%%

def NNMatFac(Data,nComp):
    from sklearn.decomposition import NMF 
    nmf = NMF(n_components=nComp,max_iter=500)
    W = nmf.fit_transform(Data)
    return W

#%%

def MeanCorrMatrix(Data,current_path):
    '''
    

    Parameters
    ----------
    Data : 3D Array or Dataframe
        [subject,psd,roi].

    Returns
    -------
    None.

    '''

    ROI=['bankssts_left',
    'bankssts_right',
    'caudalanteriorcingulate_left',
    'caudalanteriorcingulate_right',
    'caudalmiddlefrontal_left',
    'caudalmiddlefrontal_right',
    'cuneus_left',
    'cuneus_right',
    'entorhinal_left',
    'entorhinal_right',
    'frontalpole_left',
    'frontalpole_right',
    'fusiform_left',
    'fusiform_right',
    'inferiorparietal_left',
    'inferiorparietal_right',
    'inferiortemporal_left',
    'inferiortemporal_right',
    'insula_left',
    'insula_right',
    'isthmuscingulate_left',
    'isthmuscingulate_right',
    'lateraloccipital_left',
    'lateraloccipital_right',
    'lateralorbitofrontal_left',
    'lateralorbitofrontal_right',
    'lingual_left',
    'lingual_right',
    'medialorbitofrontal_left',
    'medialorbitofrontal_right',
    'middletemporal_left',
    'middletemporal_right',
    'paracentral_left',
    'paracentral_right',
    'parahippocampal_left',
    'parahippocampal_right',
    'parsopercularis_left',
    'parsopercularis_right',
    'parsorbitalis_left',
    'parsorbitalis_right',
    'parstriangularis_left',
    'parstriangularis_right',
    'pericalcarine_left',
    'pericalcarine_right',
    'postcentral_left',
    'postcentral_right',
    'posteriorcingulate_left',
    'posteriorcingulate_right',
    'precentral_left',
    'precentral_right',
    'precuneus_left',
    'precuneus_right',
    'rostralanteriorcingulate_left',
    'rostralanteriorcingulate_right',
    'rostralmiddlefrontal_left',
    'rostralmiddlefrontal_right',
    'superiorfrontal_left',
    'superiorfrontal_right',
    'superiorparietal_left',
    'superiorparietal_right',
    'superiortemporal_left',
    'superiortemporal_right',
    'supramarginal_left',
    'supramarginal_right',
    'temporalpole_left',
    'temporalpole_right',
    'transversetemporal_left',
    'transversetemporal_right']
    
    dka=pd.read_csv(current_path+'/example4Luc 4/dka_data.csv')
    corr_matrix=[]
    for sub in range(Data.shape[0]):
        df=pd.DataFrame(Data[sub,:,:], columns=ROI)
        if sub == 0:
            corr_matrix = df.corr()
            continue
        corr_matrix+=df.corr()
    corr_matrix/=sub+1
    plt.figure()
    sns.heatmap(corr_matrix)
    plt.title('Correlation Matrix [0:50] Hz')
    plt.show()
    
    for col in corr_matrix.columns:
        dka[col]=np.array(corr_matrix[col])
        
    dka.to_csv(current_path+'/example4Luc 4/dka_correlation.csv',index=False)
    
#%% 
def PlotErrorvsAge(meanPred,meanError,labels,Age,Catell_noNan,ExpectedCatell):
    RoundAge=copy.copy(labels.astype(str))
    RoundAge[labels<45]='18-45'
    RoundAge[labels>=65]='65+'
    RoundAge[np.logical_and(labels>=45, labels<65)]='45-65'
    resCatell=Catell_noNan-ExpectedCatell
    dfError=pd.DataFrame({'Mean Error': meanError,
                          'Catell': Catell_noNan,
                          'ECatell':ExpectedCatell,
                          'Catell Residuals':resCatell,
                          'Age': labels,
                          'Intervals':RoundAge})
    sns.relplot(data=dfError,y='Catell Residuals',x='Mean Error',hue='Age',palette="rocket")
    rsq,pvalue=scipy.stats.pearsonr(meanError,resCatell)
    mE=meanError.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(mE, resCatell)
    # Predict data of estimated models
    line_X = np.linspace(mE.min(), mE.max(),603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR= '+str(round(rsq,2)),
                 (10,10),fontsize=12)
    # plt.annotate('pvalue= '+str(pvalue),
    #              (20,12),fontsize=12)

    plt.title('Error on Predicted Age vs Catell Score residuals')
    plt.figure()
    sns.boxplot(data=dfError,y='Mean Error',x='Intervals',palette="rocket")
    plt.title('Boxplot Error Bias per Group')

    df=pd.DataFrame({'Mean Pred Age': meanPred,
                     'Age': labels})
    sns.relplot(data=df,x='Age',y='Mean Pred Age',palette="rocket")
    rsq,pvalue=scipy.stats.pearsonr(labels,meanPred)
    label=labels.reshape(-1,1)
    mp=meanPred.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(label,meanPred)
    linRegOnPred = linear_model.LinearRegression()
    linRegOnPred.fit(mp,meanPred)
    # Predict data of estimated models
    line_X = np.linspace(label.min(), label.max(),603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    line_y2=linRegOnPred.predict(line_X)
    plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7, label='Pred vs Real Reg')
    plt.plot(line_X,line_y2,color="red", linewidth=4, alpha=.7, label='Linear Reg')
    plt.legend()
    # plt.annotate('PearsonR= '+str(round(rsq,2)),
    #              (50,30),fontsize=12)
    # plt.annotate('pvalue= '+str(pvalue),
    #              (20,12),fontsize=12)

    plt.title('Mean predicted Age vs Chronological Age')