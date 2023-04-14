#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:29:02 2023

@author: isaac
"""
plt.figure(figsize=(10,8))
boxplot=sns.scatterplot(data=AlphaBetaCoeffDf,x='alphapws',y='betapws',hue='Cohort')
boxplot.axes.set_title("Scatterplot, Efect Size = 0.8, N=all, Time=12 seconds per step",fontsize=20)
boxplot.set_xlabel("AlphaPws",fontsize=15)
boxplot.set_ylabel("BetaPws",fontsize=15)
boxplot.tick_params(labelsize=12)
plt.figure(figsize=(10,8))
boxplot=sns.displot(data=AlphaBetaCoeffDf,x='alphapws',y='betapws',hue='Cohort',kind='kde')
# plt.title("F1 Score BoxPlot, Efect Size = 0.8, N=all, Time=12 seconds per step",fontsize=20)
plt.xlabel("AlphaPws",fontsize=15)
plt.ylabel("AlphaPws",fontsize=15)
plt.tick_params(labelsize=12)

#%%
plt.figure(figsize=(10,8))
plt.imshow(MeanF1,aspect='auto')
plt.colorbar()