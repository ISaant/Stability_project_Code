#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:56:23 2023

@author: isaac
"""

import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
current_path = os.getcwd()
sns.set(font_scale=1)
data=pd.read_csv(current_path+'/example4Luc 4/dka_data_Catell.csv')
data.rename(columns={'Acc':'Decoding Corr','network':'YEO'},inplace=True)
data['Decoding Corr']=MeanDiffJustOne2
YEO=np.repeat(data['YEO'].to_list(),100)
df=pd.DataFrame({'YEO':YEO, 'Decoding Corr':MeanDiffJustOneMatrix})
sns.boxplot(data,x='YEO',y='Decoding Corr',palette="viridis").set(title='YEO decoding Correlation Boxplot')#gist_earth_r, mako_r, rocket_r
plt.figure()
plt.xticks(rotation=15, ha='right')
sns.set_context("talk",font_scale=1.1) 
sns.violinplot(df,x='YEO',y='Decoding Corr',palette="viridis", size=3).set(title='YEO decoding Correlation')
# sns.swarmplot(df,x='YEO',y='Decoding Corr',color="white", edgecolor="gray")

plt.xticks(rotation=15, ha='right')