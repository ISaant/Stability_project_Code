#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 22:05:02 2023

@author: isaac
"""
sns.set(font_scale=1)
data=pd.read_csv(current_path+'/example4Luc 4/dka_data.csv')
data.rename(columns={'Acc':'Decoding Corr','network':'YEO'},inplace=True)
sns.boxplot(data,x='YEO',y='Decoding Corr',palette="viridis").set(title='YEO decoding Correlation Boxplot')#gist_earth_r, mako_r, rocket_r
plt.xticks(rotation=15, ha='right')