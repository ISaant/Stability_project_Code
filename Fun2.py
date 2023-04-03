#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:53:11 2023

@author: isaac
"""

import numpy as np
import pandas as pd
import os 
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
        self.freqs=freqs
     # def 
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
Data.Generate_Window_idx(10, 10)
Data.readCSV_and_Append()

DataFrame=Data.Dataframe
freqs=Data.freqs
columns=Data.columns
main=Data.main
idx=Data.idx

index,count=np.unique(DataFrame.index,return_counts=True)
Git().git_pull()
Git().git_push()


    