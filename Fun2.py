#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:53:11 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import os 
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from fooof import FOOOF
from git import Repo



current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path, os.pardir))
PATH_OF_GIT_REPO = current_path+'/.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'comment from python script'

class Generate_Data:
    
    band='alpha'
    case=0.8
    in_between=[1,40]
    def __init__(self):
        #Run Validation to the recived arguments
        
        assert isinstance(Generate_Data.band, str), f"Band value {Generate_Data.band} must be string"
        assert Generate_Data.case == 0.2 or Generate_Data.case == 0.5 or Generate_Data.case == 0.8, "Posible cases are 0.2, 0.5 or 0.8"
        assert isinstance(Generate_Data.in_between, list),  "Between must be a list with 2 values [low freq, top freq]"
        assert np.array(Generate_Data.in_between).shape[0] == 2, "Between must be a list of len=2 with values [low freq, top freq]" 
        
        #Assign to self object
        self.path = ParentPath+'/Stability-project_db/simulated/'+Generate_Data.band+'/'
        
    def get_path (self):
        print(self.path, '  +  case='+str(self.case)+'_250_PSD')
    
    def OpenCase (self):
        casePath=self.path+str(Generate_Data.case)+'_250_PSD'
        Dir=np.sort(os.listdir(casePath))[2:-2]
        self.Dir=Dir

    def Generate_Window_idx (self, Windows=10,sampleSize=250):
        #Run Validation to the recived arguments
        assert isinstance(Windows, int) and Windows <=100, "Windows must be a single integer between 1 and 100" 
        assert isinstance(sampleSize, int) and sampleSize <= 250, "sampleSize must be a single integer between 1 and 250"
        # seed=np.random.seed(21)
        self.timeWindows=np.random.choice(self.Dir,size=Windows,replace=False) #iterar esto primero
        self.idx=np.sort(np.random.choice(np.arange(0,250),size=sampleSize,replace=False))
        #
        # print(self.OpenCase())
        
        # return seed, self.timeWindows, self.idx
        
    def readCSV_and_Append (self):
        Dataframe=pd.DataFrame()
        freqs=np.linspace(0,250,(250*3)+1,endpoint=True)
        columns= [i for i, x in enumerate((freqs>=self.in_between[0]) & (freqs<self.in_between[1])) if x]
        for win in tqdm(self.timeWindows):
            main=pd.read_csv(self.path+str(self.case)+'_250_PSD/'+win,header=None)
            self.main=main[main.columns[columns]].iloc[np.concatenate((self.idx,self.idx+250))]
            Dataframe=pd.concat([Dataframe,self.main],ignore_index=False)
        Dataframe.reset_index(inplace=True)
        Dataframe.rename({'index':'id'},axis=1,inplace=True)
        Dataframe=Dataframe.groupby('id').mean()
        Dataframe['Cohort']=np.concatenate((np.zeros(len(self.idx)),np.ones(len(self.idx)))).astype(int)
        self.Dataframe=Dataframe
        self.columns=columns
        self.freqs=freqs[columns]
     # def 
     
class Periodic_Aperiodic:
    
    def __init__(self):
        self.periodic=pd.DataFrame()
        self.aperiodic=pd.DataFrame()
    
    def fooof(self,Data):
        inBetween=Data.in_between
        df=copy.copy(Data.Dataframe)
        cohort=df['Cohort']
        df.drop(columns=['Cohort'],inplace=True)
        periodic= copy.copy(df)
        aperiodic=copy.copy(df)
        parameters=[]
        for i in tqdm(range(df.shape[0])):
            fm = FOOOF(max_n_peaks=1, aperiodic_mode='fixed')
            fm.add_data(Data.freqs, np.array(df.iloc[i]),inBetween)
            fm.fit(freqs, np.array(df.iloc[i]), inBetween)
            periodic.iloc[i]=fm._peak_fit
            aperiodic.iloc[i]=fm._ap_fit
            exp = fm.get_params('aperiodic_params', 'exponent')
            offset = fm.get_params('aperiodic_params', 'offset')
            cfs = fm.get_params('peak_params', 'CF')
            pws = fm.get_params('peak_params', 'PW')
            bws = fm.get_params('peak_params', 'BW')
            parameters.append([exp,offset,cfs,pws,bws])
        periodic['Cohort']=cohort
        aperiodic['Cohort']=cohort
        parameters=np.array(parameters)
        parameters=np.concatenate((parameters,np.array([cohort]).T),axis=1)
        parameters=pd.DataFrame(parameters, columns=['exp','offset','cfs','pws','bws','Cohort'])
        self.periodic=periodic
        self.aperiodic=aperiodic
        self.parameters=parameters
            
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
        plt.plot(freqs,meanA,'r')
        plt.plot(freqs,meanB,'g')
        plt.fill_between(freqs,meanA+stdA,meanA-stdA,alpha=.5,color='r')
        plt.fill_between(freqs,meanB+stdB,meanB-stdB,alpha=.5,color='g')
        plt.show()
    
    def boxplot_coeffs (self):
        Par=self.parameters
        fig, ax = plt.subplots(1, 5, figsize=(10, 6))
        sns.boxplot( x=Par['Cohort'], y=Par['exp'],ax=ax[0] )
        sns.boxplot( x=Par['Cohort'], y=Par['offset'],ax=ax[1] )
        sns.boxplot( x=Par['Cohort'], y=Par['cfs'],ax=ax[2] )
        sns.boxplot( x=Par['Cohort'], y=Par['pws'],ax=ax[3] )
        sns.boxplot( x=Par['Cohort'], y=Par['bws'],ax=ax[4] )
        plt.subplots_adjust(wspace=0.5) 

        plt.show()
        
        
        
        
    # def get_df(self,Data):
    #     return Data.Dataframe

class Git:
    def __init__ (self):
        self.git_path=PATH_OF_GIT_REPO
        self.message=COMMIT_MESSAGE
    def git_push(self):
        try:
            repo = Repo(self.git_path)
            # repo.git.add(update=True)
            repo.git.add(all=True)
            repo.index.commit(self.message)
            origin = repo.remote(name='origin')
            origin.push()
        except:
            print('Some error occured while pushing the code')    

    def git_pull(self):
        try:
            repo = Repo(self.git_path)
            origin = repo.remote(name='origin')
            origin.pull()
        except:
            print('Some error occured while pulling the code')           
# class Generate_Data (Paths):
#     inBetween=[1,40]
#     def __init__(self):
#         super().__init__(self)
#         self.PSD=[]
    
    # def add_PSD:
    #     pass
        # self.PSD.append()
    
# timeWindows=np.random.choice(Dir,size=50,replace=False) #iterar esto primero
# idx=np.sort(np.random.choice(np.arange(0,250),size=250,replace=False))

# case1=Generate_Data()
# Dir1=case1.OpenCase()
# case1.Generate_Window_idx()

# Generate_Data.band='alpha'
# Generate_Data.in_between=[1]
Data=Generate_Data()
Data.get_path()
Data.OpenCase()
Data.Generate_Window_idx(Windows=50, sampleSize=250)
Data.readCSV_and_Append()

DataFrame=Data.Dataframe
freqs=Data.freqs
columns=Data.columns
main=Data.main
idx=Data.idx

index,count=np.unique(DataFrame.index,return_counts=True)

APer=Periodic_Aperiodic()
APer.fooof(Data)
Per=APer.get_Periodic()
Ap=APer.get_Aperiodic()
Par=APer.get_Parameters()
APer.plot_parameters('periodic')
APer.plot_parameters('aperiodic')
APer.boxplot_coeffs()
Git().git_pull()
Git().git_push()


    