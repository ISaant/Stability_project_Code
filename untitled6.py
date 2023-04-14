#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:18:10 2023

@author: isaac
"""

APer_Alpha_1peak = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=i, 
                      plot=True, in_between=[1,50],
                      max_n_peaks=1, fit='fixed')

APer_Beta_1peak = GetStuff('beta', Windows=99, 
                      sampleSize=250,seed=i, 
                      plot=True, in_between=[1,50],
                      max_n_peaks=1, fit='fixed')

boxplot=sns.boxplot(a, showmeans=True,
            meanprops={'marker':'o',
                       'markerfacecolor':'white', 
                       'markeredgecolor':'black',
                       'markersize':'8'})

boxplot.axes.set_title("F1 Score BoxPlot, Efect Size = 0.8, N=all, Time=12 seconds per step",fontsize=20)
boxplot.set_xlabel("Windows",fontsize=15)
boxplot.set_ylabel("F1 Score",fontsize=15)
boxplot.tick_params(labelsize=12)