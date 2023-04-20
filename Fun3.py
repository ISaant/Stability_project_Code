#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:55:24 2023

@author: sflores
"""

import numpy as np
import pandas as pd
import os 
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from fooof import FOOOF
from scipy.stats import ttest_ind
from scipy.fftpack import dct

current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path, os.pardir))

class Generate_Data:
    
    band='alpha'
    effectSize=0.8
    in_between=[1,40]
    def __init__(self):
        #Run Validation to the recived arguments
        
        assert isinstance(Generate_Data.band, str), f"Band value {Generate_Data.band} must be string"
        assert Generate_Data.effectSize == 0.2 or Generate_Data.effectSize == 0.5 or Generate_Data.effectSize == 0.8, "Posible effectSizes are 0.2, 0.5 or 0.8"
        assert isinstance(Generate_Data.in_between, list),  "Between must be a list with 2 values [low freq, top freq]"
        assert np.array(Generate_Data.in_between).shape[0] == 2, "Between must be a list of len=2 with values [low freq, top freq]" 
        
        #Assign to self object
        self.path = ParentPath+'/Stability-project_db/PSDfromTimeSeries_MultipleFeatures/'
        
    def get_path (self):
        print(self.path, '  +  effectSize='+str(self.effectSize)+'_250_PSD')
    
    def Open_effectSize (self):
        effectSizePath=self.path+str(Generate_Data.effectSize)+'_250_PSD'
        Dir=np.sort(os.listdir(effectSizePath))
        self.Dir=Dir

    def Generate_Window_idx (self, Windows=10,sampleSize=250,seed=21):
        #Run Validation to the recived arguments
        assert isinstance(Windows, int) and Windows <=99, "Windows must be a single integer between 1 and 99" 
        assert isinstance(sampleSize, int) and sampleSize <= 250, "sampleSize must be a single integer between 1 and 250"
        seed=np.random.seed(seed)
        self.timeWindows=np.random.choice(self.Dir,size=Windows,replace=False) #iterar esto primero
        self.idx=np.sort(np.random.choice(np.arange(0,250),size=sampleSize,replace=False))
        #
        # print(self.OpeneffectSize())
        
        # return seed, self.timeWindows, self.idx
        
    def readCSV_and_Append (self):
        Dataframe=pd.DataFrame()
        Dataframe_full=pd.DataFrame()
        freqs=np.linspace(0,250,(250*3)+1,endpoint=True)
        columns= [i for i, x in enumerate((freqs>=self.in_between[0]) & (freqs<self.in_between[1])) if x]
        for win in tqdm(self.timeWindows):
            main=pd.read_csv(self.path+str(self.effectSize)+'_250_PSD/'+win,header=None)
            Dataframe_full=pd.concat([Dataframe_full,main],ignore_index=False)
            self.main=main[main.columns[columns]].iloc[np.concatenate((self.idx,self.idx+250))]
            Dataframe=pd.concat([Dataframe,self.main],ignore_index=False)
        Dataframe.reset_index(inplace=True)
        Dataframe.rename({'index':'id'},axis=1,inplace=True)
        Dataframe=Dataframe.groupby('id').mean()
        Dataframe['Cohort']=np.concatenate((np.zeros(len(self.idx)),np.ones(len(self.idx)))).astype(int)
        Dataframe_full.reset_index(inplace=True)
        Dataframe_full.rename({'index':'id'},axis=1,inplace=True)
        Dataframe_full=Dataframe_full.groupby('id').mean()
        Dataframe_full['Cohort']=np.concatenate((np.zeros(250),np.ones(250))).astype(int)
        self.Dataframe=Dataframe
        self.Dataframe_full=Dataframe_full
        self.columns=columns
        self.freqs=freqs[columns]
        self.freqs_full=freqs
    
    def plot_timeSeries(self):
        freqs=np.linspace(0,250,(250*3)+1,endpoint=True)
        Dataframe=pd.DataFrame()
        columns= [i for i, x in enumerate((freqs>=self.in_between[0]) & (freqs<self.in_between[1])) if x]
        freqs=freqs[columns]
        for file in self.Dir:
            main=pd.read_csv(self.path+str(self.effectSize)+'_250_PSD/'+file,header=None)
            main=main[main.columns[columns]]
            Dataframe=pd.concat([Dataframe,main],ignore_index=False)
        Dataframe.reset_index(inplace=True)
        Dataframe.rename({'index':'id'},axis=1,inplace=True)
        Dataframe=Dataframe.groupby('id').mean()

        PSD=np.array(Dataframe)
        plt.figure()
        meanA=np.mean(PSD[:int(PSD.shape[0]/2)],axis=0)
        meanB=np.mean(PSD[int(PSD.shape[0]/2):],axis=0)
        stdA=np.std(PSD[:int(PSD.shape[0]/2)],axis=0)
        stdB=np.std(PSD[int(PSD.shape[0]/2):],axis=0)
        # plt.plot(np.log(freqs),np.log(meanA),'r')
        # plt.plot(np.log(freqs),np.log(meanB),'g')
        print(np.sqrt(PSD.shape[0]/2))
        coefIntervalA=(1.96*(stdA)/np.sqrt(PSD.shape[0]/2))
        coefIntervalB=(1.96*(stdB)/np.sqrt(PSD.shape[0]/2))
        # plt.fill_between(np.log(freqs),np.log(meanA+coefIntervalA),np.log(meanA-coefIntervalA),alpha=.5,color='r')
        # plt.fill_between(np.log(freqs),np.log(meanB+coefIntervalB),np.log(meanB-coefIntervalB),alpha=.5,color='g')
        plt.plot(freqs,meanA,'r')
        plt.plot(freqs,meanB,'g')
        plt.fill_between(freqs,meanA+coefIntervalA,meanA-coefIntervalA,alpha=.5,color='r')
        plt.fill_between(freqs,meanB+coefIntervalB,meanB-coefIntervalB,alpha=.5,color='g')
        
        plt.title('Band= '+Generate_Data.band+', Mean A vs B groups, all widows, all sujects. Freqs between'+str(self.in_between))
        plt.show()
            
     
class Periodic_Aperiodic:
    
    def __init__(self):
        self.periodic=pd.DataFrame()
        self.aperiodic=pd.DataFrame()
        self.freqs=[]
        self.max_n_peaks=6
        self.fit='fixed'
    def fooof(self,Data):
        inBetween=Data.in_between
        df=copy.copy(Data.Dataframe)
        cohort=df['Cohort']
        df.drop(columns=['Cohort'],inplace=True)
        periodic= copy.copy(df)
        aperiodic=copy.copy(df)
        whitened=copy.copy(df)
        self.freqs=Data.freqs
        self.columns=Data.columns
        self.NFFT=Data.Dataframe_full.to_numpy()[:,:-1].shape[1]*2
        parameters=[]
        for i in tqdm(range(df.shape[0])):
            fm = FOOOF(max_n_peaks=self.max_n_peaks, aperiodic_mode=self.fit)
            fm.add_data(self.freqs, np.array(df.iloc[i]),inBetween)
            fm.fit(self.freqs, np.array(df.iloc[i]), inBetween)
            periodic.iloc[i]=fm._peak_fit
            aperiodic.iloc[i]=fm._ap_fit
            w=fm.power_spectrum-fm._ap_fit
            whitened.iloc[i]=w+abs(min(w))
            exp = fm.get_params('aperiodic_params', 'exponent')
            offset = fm.get_params('aperiodic_params', 'offset')
            cfs = fm.get_params('peak_params', 'CF')
            pws = fm.get_params('peak_params', 'PW')
            bws = fm.get_params('peak_params', 'BW')
            parameters.append([exp,offset,cfs,pws,bws])
            # parameters.append([cfs,pws,bws])
        periodic['Cohort']=cohort
        aperiodic['Cohort']=cohort
        parameters=np.array(parameters)
        parameters=np.concatenate((parameters,np.array([cohort]).T),axis=1)
        band=Generate_Data.band
        parameters=pd.DataFrame(parameters, columns=[band+'exp',band+'offset',band+'cfs',band+'pws',band+'bws','Cohort'])
        # parameters=pd.DataFrame(parameters, columns=[band+'cfs',band+'pws',band+'bws','Cohort'])

        self.periodic=periodic
        self.aperiodic=aperiodic
        self.parameters=parameters
        self.whitened=whitened
            
    def get_Periodic(self):
        return self.periodic
    
    def get_Aperiodic(self):
        return self.aperiodic
    
    def get_Parameters(self):
        return self.parameters
    
    def plot_parameters (self, component="periodic"):
        assert isinstance(component, str), "component must be string: 'periodic' or 'aperiodic'"
        x=eval("self."+component)
        plt.figure()
        PSD=np.array(x.iloc[:,0:-1])
        meanA=np.mean(PSD[:int(PSD.shape[0]/2)],axis=0)
        meanB=np.mean(PSD[int(PSD.shape[0]/2):],axis=0)
        stdA=np.std(PSD[:int(PSD.shape[0]/2)],axis=0)
        stdB=np.std(PSD[int(PSD.shape[0]/2):],axis=0)
        # for row in range(int(PSD.shape[0]/2)):
        #     plt.plot(self.freqs,PSD[row],'r',alpha=.8)
        #     plt.plot(self.freqs,PSD[row+int(PSD.shape[0]/2)],'g',alpha=.8)
       
        plt.plot(self.freqs,meanA,'r')
        plt.plot(self.freqs,meanB,'g')
        plt.fill_between(self.freqs,meanA+stdA,meanA-stdA,alpha=.5,color='r')
        plt.fill_between(self.freqs,meanB+stdB,meanB-stdB,alpha=.5,color='g')
        plt.title('Band= '+Generate_Data.band+', '+component+' Components')
        plt.show()
    
    def boxplot_coeffs (self):
        Par=self.parameters
        band=Generate_Data.band
        fig, ax = plt.subplots(1, 5, figsize=(10, 6))
        sns.boxplot( x=Par['Cohort'], y=Par[band+'exp'],ax=ax[0] )
        sns.boxplot( x=Par['Cohort'], y=Par[band+'offset'],ax=ax[1] )
        sns.boxplot( x=Par['Cohort'], y=Par[band+'cfs'],ax=ax[2] )
        sns.boxplot( x=Par['Cohort'], y=Par[band+'pws'],ax=ax[3] )
        sns.boxplot( x=Par['Cohort'], y=Par[band+'bws'],ax=ax[4] )
        # plt.subplots_adjust(wspace=0.5) 
        plt.suptitle('Aperiodic and Periodic coeff, band=' +Generate_Data.band)
        plt.show()
        
    def scatter3D (self):
        Par=self.parameters
        band=Generate_Data.band
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        x=np.array(Par.loc[:,band+'cfs'])
        y=np.array(Par.loc[:,band+'pws'])
        z=np.array(Par.loc[:,band+'bws'])
        ax.set_xlabel(band+'cfs')
        ax.set_ylabel(band+'pws')
        ax.set_zlabel(band+'bws')
        color=Par.loc[:,'Cohort']
        ax.scatter(x,y,z,c=color,alpha=.5)
        plt.suptitle('Periodic coeff, band=' +Generate_Data.band)
        plt.show()
        
    def scatterMatrix(self):
        Par=self.parameters
        band=Generate_Data.band
        drop=[i for i,x in enumerate(Par.isna().any(axis=1)) if x]
        Par.drop(drop,inplace=True)
        band=Generate_Data.band
        def pos(colNum):
            p=np.zeros((colNum,2))
            cx=.9/(colNum+.5) 
            cy=1/colNum
            x=.2
            y=.9
            for i in range(colNum):
                p[i]=[x,y]
                x+=cx
                y-=cy
            return p
        p=pos(len(Par.columns[:-1]))
        g=sns.pairplot(Par, hue="Cohort",corner=True)
        g.map_lower(sns.kdeplot, levels=6, color=".2")
        
        def cohen_d(x,y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

        
        for i, col in enumerate(Par.columns[:-1]):
            ttest=ttest_ind(Par.loc[:250,col],Par.loc[250:,col])[1]
            cohend=cohen_d(Par.loc[:250,col],Par.loc[250:,col])

            # print(col, cohend)
            g.fig.text(p[i][0], p[i][1],'p= '+str(round(ttest,5)), fontsize=15, fontweight='bold')
        
        plt.show()
        
    def MFCCs (self,nfilt=128,num_ceps=34,sig_lift=True):
        pow_frames=self.whitened.to_numpy()
        col=self.columns
        NFFT=self.NFFT
        sample_rate=500
        #Intentar implementar self.class=getArgumentValue para saber si el usuario 
            #especifica frecuecias o fs            
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        # print(hz_points)
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)
        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 ))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
        fbank=fbank[:,col]
        plt.figure()
        plt.plot(self.freqs.T,fbank.T)
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        # filter_banks = 20 * np.log10(filter_banks)  # dB
        #Sinusoidal Lifting, Ref: https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf, 
        #                           https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/260_bandpass%20liftering.pdf
        if num_ceps=='all':
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 0 : (nfilt + 1)] 
        else: 
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 0 : (num_ceps + 1)] 
            filter_banks= filter_banks[:, 1 : (num_ceps + 1)]
         
        (nframes, ncoeff) = mfcc.shape
        cep_lifter = 22
        n = np.arange(ncoeff)
        if sig_lift=='True':
            lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
            mfcc *= lift  #*
            
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

            # img=librosa.display.specshow(mfcc.T, sr=sample_rate, x_axis='time', y_axis='linear')
        img=ax.imshow(filter_banks,aspect='auto',cmap='jet')
        ax.set_title('filter_bank')
        fig.colorbar(img, ax=ax)
        plt.show()
        self.mfcc = mfcc
        self.filter_banks=filter_banks
    
    def get_MFCCs(self):
        return self.mfcc, self.filter_banks
        
    # def get_df(self,Data):
    #     return Data.Dataframe
def getArgumentValue(argument,defaultValue,args):
    
    
    for k in range(len(args)):
       if isinstance(args[k], str):
          if argument.lower() == args[k].lower():
             break
    # print(k)     
       
    if k!=len(args)-1:
        value = args[k+1]; # Return the value, following the ARGUMENT string.
    else:
        value = defaultValue;
    
    return value

def GetStuff(band,Windows,sampleSize,seed,plot,in_between,max_n_peaks,fit):     
    Generate_Data.band=band
    Generate_Data.in_between=in_between
    Data=Generate_Data()
    # Data.get_path()
    Data.Open_effectSize()
    Data.Generate_Window_idx(Windows=Windows, sampleSize=sampleSize, seed=seed)
    Data.readCSV_and_Append()
    APer=Periodic_Aperiodic()
    APer.max_n_peaks=max_n_peaks
    APer.fit=fit
    APer.fooof(Data)
    APer.MFCCs()
    # Per=APer.get_Periodic()
    # Ap=APer.get_Aperiodic()
    # Par=APer.get_Parameters()
    # MFCCs,filter_banks=APer.get_MFCCs()
    if plot:
        # plt.close('all')
        Data.plot_timeSeries()
        # APer.plot_parameters('periodic')
        # APer.plot_parameters('aperiodic')
        # APer.boxplot_coeffs()
        # APer.scatter3D()
        # APer.scatterMatrix()
    return APer



# Data_Beta,APer_Beta,Per_Beta,Ap_Beta, Par_Beta = GetStuff('beta', Windows=99, sampleSize=250,seed=2, plot=False)
# AlphaBetaCoeffDf=pd.concat([Par_Alpha.iloc[:,:-1],Par_Beta],axis=1)

#clean nans
# drop=[i for i,x in enumerate(AlphaBetaCoeffDf.isna().any(axis=1)) if x]
# AlphaBetaCoeffDf.drop(drop,inplace=True)



    