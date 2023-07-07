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
Data=copy.copy(restStateNoOffset)
Sub,PSD,ROI=Data.shape
fig = figure(figsize=(10,7))
plt.suptitle('PSD per Age Group')
Data=Data[Age.argsort(),:,:]
AgeSorted=np.sort(Age)
AgeStep=2.5
SubGroup=np.arange(min(Age)+(AgeStep*2),max(Age)-AgeStep*2,AgeStep)
colors=linear_gradient('#ffd89b', '#19547b', np.floor(len(SubGroup)-1).astype(int)+1)
grid = plt.GridSpec(2, 12, wspace=0.4, hspace=0.3)
freqRange1=np.where((freqs>=5)*(freqs<=35))[0]
freqRange2=np.where((freqs>49)*(freqs<51))[0]
for i,s in enumerate(tqdm(SubGroup)):
    # s=int(np.round(s))
    print(i)
    AgeRange=np.where((AgeSorted>=s)*(AgeSorted<s+AgeStep))[0]
    mean=np.mean(np.mean(Data[AgeRange,:,:],axis=2),axis=0)
    # meanAge=np.round(np.mean(Age[s:s+SubGroup]))
    plt.subplot(grid[0, :])
   
    # plot(np.log(freqs[:PSD]),np.log(mean),color=colors['hex'][i], label = str(s+AgeStep))
    # plt.subplot(grid[1, 3:10])
    # plot(np.log(freqs[freqRange1]),np.log(mean[freqRange1]),color=colors['hex'][i])
    # plt.subplot(grid[1, 10:])
    # plot(np.log(freqs[freqRange2]),np.log(mean[freqRange2]),color=colors['hex'][i])
    # plt.yticks([])
    plot(freqs[:PSD],mean,color=colors['hex'][i], label = str(s+AgeStep))
    plt.subplot(grid[1, 3:10])
    plot(freqs[freqRange1],mean[freqRange1],color=colors['hex'][i])
    plt.subplot(grid[1, 10:])
    plot(freqs[freqRange2],mean[freqRange2],color=colors['hex'][i])
    plt.yticks([])
    if i == 0:
        Mean=mean
        continue
    Mean+=mean
Mean/=(i+1)
plt.subplot(grid[0, :])
plot(freqs[:PSD],Mean,'k',label='mean')
# plt.title('freqs [0:150]')
plt.xlabel('log(Frequencies  [0:150] Hz)')
plt.ylabel('log(Power)')
plt.legend()
plt.subplot(grid[1, 3:10])
plot(freqs[freqRange1],Mean[freqRange1],'k',)
plt.xlabel('log(Frequencies [5:35]Hz)')
plt.ylabel('log(Power)')
plt.subplot(grid[1, 10:])
plot(freqs[freqRange2],Mean[freqRange2],'k',)
plt.xlabel('log(Freq [49:51]Hz)')
# plot(freqs[:PSD],Mean,'k',label='mean')
# plt.subplot(grid[:, 3])
