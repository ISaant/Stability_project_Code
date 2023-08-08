#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:16:22 2023

@author: sflores
"""
Data=pAD_PSD
Sub,PSD,ROI=Data.shape
# columnsInBetween= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1]+.5)) if x]
figure()
sub=2
psd_mean=np.mean(Data[sub,4:80,:],axis=1)

for roi in tqdm(range(ROI)):
    psd=Data[sub,4:80,roi]
    # plot(np.log(freqs),np.log(mean),alpha=.2)
    plot(freqs[4:80],psd,alpha=0.2)

plt.plot(freqs[4:80],psd_mean,'k')

plt.figure()
# columnsInBetween= [i for i, x in enumerate((freqs>=inBetween[0]) & (freqs<inBetween[1]+.5)) if x]
figure()
for sub in tqdm(range(Sub)):
    mean=np.mean(Data[sub,:,:],axis=1)
    # plot(np.log(freqs),np.log(mean),alpha=.2)
    # plot(np.log(freqs),np.log(mean))
    plot(freqs[4:80],mean[4:80],alpha=.2)

    # plt.show()
    # plt.pause(1)

    if sub == 0:
        Mean=mean
        continue
    Mean+=mean
Mean/=(Sub)
# plot(np.log(freqs),np.log(Mean),'k')
# plot(np.log(freqs),np.log(Mean),'k')
plot(freqs[4:80],Mean[4:80],'k')


plt.title('Global PSD')
plt.xlabel('log(Frequencies [Hz])')
plt.ylabel('log(Power)')
