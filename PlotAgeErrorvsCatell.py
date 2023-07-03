#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:28:56 2023

@author: sflores
"""
import seaborn as sns

RoundAge=copy.copy(labels.astype(str))
RoundAge[labels<45]='18-45'
RoundAge[labels>=65]='65+'
RoundAge[np.logical_and(labels>=45, labels<65)]='45-65'

dfError=pd.DataFrame({'Mean Error': meanError,
                      'Catell': Catell_noNan,
                      'Age': labels,
                      'Intervals':RoundAge})
sns.relplot(data=dfError,y='Catell',x='Mean Error',hue='Age',palette="rocket")
rsq,pvalue=scipy.stats.pearsonr(meanError,Catell_noNan)
mE=meanError.reshape(-1,1)
linReg = linear_model.LinearRegression()
linReg.fit(mE, Catell_noNan)
# Predict data of estimated models
line_X = np.linspace(mE.min()-5, mE.max()+5,603)[:, np.newaxis]
line_y = linReg.predict(line_X)
plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
             (10,40),fontsize=12)
# plt.annotate('pvalue= '+str(pvalue),
#              (20,12),fontsize=12)

plt.title('Error on Predicted age vs Catell Score')
plt.figure()
sns.boxplot(data=dfError,y='Mean Error',x='Intervals',palette="rocket")

#%%
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
line_X = np.round(np.arange(label.min()-5, label.max()+5,.01),2)[:, np.newaxis]
line_y = linReg.predict(line_X)
line_y2=linRegOnPred.predict(line_X)
plt.plot(line_X,line_y,color="magenta", linewidth=4, alpha=.7)
# plt.plot(line_X,line_y2,color="red", linewidth=4, alpha=.7)

plt.annotate('PearsonR= '+str(round(rsq,2)),
             (50,30),fontsize=12)
plt.annotate('MAE= =+-7.77 years',(50,25),fontsize=11)
# plt.annotate('pvalue= '+str(pvalue),
#              (20,12),fontsize=12)

plt.title('Mean predicted age vs Age')

#%%
fig, axs = plt.subplots()
df=pd.DataFrame({'Mean Pred Age': meanPred,
                 'Age': labels})
# sns.scatterplot(data=df,x='Age',y='Mean Pred Age',palette="rocket",alpha=.2, ax=axs)
rsq,pvalue=scipy.stats.pearsonr(labels,meanPred)
label=labels.reshape(-1,1)
mp=meanPred.reshape(-1,1)
linReg = linear_model.LinearRegression()
linReg.fit(label,meanPred)
linRegOnPred = linear_model.LinearRegression()
linRegOnPred.fit(mp,meanPred)
# Predict data of estimated models
line_X = np.round(np.arange(label.min()-5, label.max()+5,.01),2)[:, np.newaxis]
line_y = linReg.predict(line_X)
line_y2=linRegOnPred.predict(line_X)
diff=np.subtract(line_y,line_y2)
# plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7,)
plt.plot(line_X,line_y2,color="red", linewidth=4, alpha=.7)
plt.annotate('PearsonR= '+str(round(rsq,2)),
             (60,30),fontsize=12)


meanPredCorrected=copy.copy(meanPred)
for i in range(len(meanPredCorrected)):
    meanPredCorrected[i]=meanPred[i]-diff[np.where(line_X==labels[i])[0]]

df=pd.DataFrame({'Mean Pred Age Corrected': meanPredCorrected,
                 'Age': labels,
                 'Catell': Catell_noNan})
sns.scatterplot(data=df,x='Age',y='Mean Pred Age Corrected',palette="rocket",ax=axs)

#%%
dfError=pd.DataFrame({'Mean Error': labels-meanPredCorrected,
                      'Catell': Catell_noNan,
                      'Age': labels,
                      'Intervals':RoundAge})
meanErrorCorrected=labels-meanPredCorrected
sns.relplot(data=dfError,y='Catell',x='Mean Error',hue='Age',palette="rocket")
rsq,pvalue=scipy.stats.pearsonr(meanErrorCorrected,Catell_noNan)
mE=meanErrorCorrected.reshape(-1,1)
linReg = linear_model.LinearRegression()
linReg.fit(mE, Catell_noNan)
# Predict data of estimated models
line_X = np.linspace(mE.min()-5, mE.max()+5,603)[:, np.newaxis]
line_y = linReg.predict(line_X)
plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7)
plt.annotate('PearsonR_sq= '+str(round(rsq,2)),
             (10,40),fontsize=12)
# plt.annotate('pvalue= '+str(pvalue),
#              (20,12),fontsize=12)

plt.title('Error on Predicted age vs Catell Score')
plt.figure()
sns.boxplot(data=dfError,y='Mean Error',x='Intervals',palette="rocket")

#%%
RoundAge=copy.copy(labels.astype(str))
RoundAge[labels<45]='18-45'
RoundAge[labels>=65]='65+'
RoundAge[np.logical_and(labels>=45, labels<65)]='45-65'
resCatell=Catell_noNan-ExpectedCatell
dfError=pd.DataFrame({'Mean Error': labels-meanPredCorrected,
                      'Catell': Catell_noNan,
                      'ECatell':ExpectedCatell,
                      'Catell Residuals':resCatell,
                      'Age': labels,
                      'Intervals':RoundAge})
meanErrorCorrected=labels-meanPredCorrected
sns.relplot(data=dfError,y='Catell Residuals',x='Mean Error',hue='Age',palette="rocket")
rsq,pvalue=scipy.stats.pearsonr(meanErrorCorrected,resCatell)
mE=meanErrorCorrected.reshape(-1,1)
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

# df=pd.DataFrame({'Mean Pred Age': meanPred,
#                  'Age': labels})
# sns.relplot(data=df,x='Age',y='Mean Pred Age',palette="rocket")
# rsq,pvalue=scipy.stats.pearsonr(labels,meanPred)
# label=labels.reshape(-1,1)
# mp=meanPred.reshape(-1,1)
# linReg = linear_model.LinearRegression()
# linReg.fit(label,meanPred)
# linRegOnPred = linear_model.LinearRegression()
# linRegOnPred.fit(mp,meanPred)
# # Predict data of estimated models
# line_X = np.linspace(label.min(), label.max(),603)[:, np.newaxis]
# line_y = linReg.predict(line_X)
# line_y2=linRegOnPred.predict(line_X)
# plt.plot(line_X,line_y,color="yellowgreen", linewidth=4, alpha=.7, label='Pred vs Real Reg')
# plt.plot(line_X,line_y2,color="red", linewidth=4, alpha=.7, label='Linear Reg')
# plt.legend()
# # plt.annotate('PearsonR= '+str(round(rsq,2)),
# #              (50,30),fontsize=12)
# # plt.annotate('pvalue= '+str(pvalue),
# #              (20,12),fontsize=12)

# plt.title('Mean predicted Age vs Chronological Age')