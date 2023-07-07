#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:33:24 2023

@author: isaac
"""

# columnsInBetween= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1]+.5)) if x]
'''
jupiter='#ffd89b', '#19547b'
sherbert='#f79d00','#64f38c'
dusk='#2c3e50','#fd746c'
grapefruit='#e96443','#904e95'
'''
Data=copy.copy(restState)
Sub,PSD,ROI=Data.shape
figure()
Data=Data[Age.argsort(),:,:]
AgeSorted=np.sort(Age)
AgeStep=5
SubGroup=np.arange(min(Age),max(Age)-AgeStep,AgeStep)
colors=linear_gradient('#2c3e50','#fd746c', np.floor(len(SubGroup)).astype(int)+1)

for i,s in enumerate(tqdm(SubGroup)):
    # s=int(np.round(s))
    print(i)
    AgeRange=np.where((AgeSorted>=s)*(AgeSorted<s+AgeStep))[0]
    mean=np.mean(np.mean(Data[AgeRange,:,:],axis=2),axis=0)
    # meanAge=np.round(np.mean(Age[s:s+SubGroup]))
    plot(np.log(freqs[:PSD]),np.log(mean),color=colors['hex'][i],alpha=.7, label = str(s+AgeStep))
    # plot(freqs[:PSD],mean,color=colors['hex'][i],alpha=.7, label = str(meanAge))

    if i == 0:
        Mean=mean
        continue
    Mean+=mean
Mean/=(i+1)
plot(np.log(freqs[:PSD]),np.log(Mean),'k',label='mean')
# plot(freqs[:PSD],Mean,'k',label='mean')

plt.title('Global PSD')
plt.xlabel('log(Frequencies [Hz])')
plt.ylabel('log(Power)')
plt.legend()