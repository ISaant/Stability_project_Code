#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:47:22 2023

@author: isaac
"""

def PlotDist(demographics):
    import copy
    from sklearn import linear_model
    
    # Bar=[sum(count[unique<30])]
    # for i in np.arange(30,90,10):
    #     Bar.append(sum(count[np.logical_and(unique>=i, unique<i+10)]))
    # plt.figure()
    sns.displot(data=demographics,x='age',kde=True)
    plt.title('Age Histogram')
    Age=demographics['age'].to_numpy()
    Catell=demographics['Catell_score']
    # unique,count=np.unique(np.round(Age),return_counts=True)
    RoundAge=copy.copy(Age)
    RoundAge[RoundAge<30]=30
    for i in np.arange(20,80,10):
        RoundAge[np.logical_and(RoundAge>=i, RoundAge<i+10)]=i
    RoundAge[RoundAge>80]=80
    demographics['Intervals']=RoundAge
    # plt.figure()
    sns.displot(data=demographics,x='Catell_score', hue='Intervals',kind='kde', fill=True)
    plt.title('Catell Score Distribution')
    sns.relplot(data=demographics,y='Catell_score', x='age', hue='newAge')
    plt.title('Age-Catell Regression')
    idx=np.argwhere(np.isnan(Catell))
    Catell=np.delete(Catell, idx)
    Age=np.delete(Age, idx)
    rsq,pvalue=scipy.stats.pearsonr(Age,Catell)
    # print(pearson)
    Age=Age.reshape(-1,1)
    linReg = linear_model.LinearRegression()
    linReg.fit(Age, Catell)
    # r_sq = ransac.score(Age, Catell)
    # r_sq=ransac.estimator_.coef_
    # print(f"coefficient of determination: {r_sq}")
    # print(f"coefficient of determination: {r_sq}")
    predLine = linReg.predict(Age)
    # print(scipy.stats.pearsonr(Age,Catell))

    # Predict data of estimated models
    line_X = np.linspace(Age.min(), Age.max(),603)[:, np.newaxis]
    line_y = linReg.predict(line_X)
    plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
    plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
                 (20,15),fontsize=12)
    plt.annotate('pvalue= '+str(pvalue),
                 (20,12),fontsize=12)
plt.close('all')
PlotDist(demographics)
plt.show()
