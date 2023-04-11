#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:36:16 2023

@author: isaac
"""

AlphaBetaCoeffDf=pd.concat([Par_Alpha.loc[:,'alphapws'],Par_Beta.loc[:,'betapws'],Par_Beta.loc[:,'Cohort']],axis=1)
sns.scatterplot(data=AlphaBetaCoeffDf,x='alphapws',y='betapws',hue='Cohort')
sns.displot(data=AlphaBetaCoeffDf,x='alphapws',y='betapws',hue='Cohort',kind='kde')