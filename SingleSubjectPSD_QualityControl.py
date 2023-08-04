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
sub=59
psd_mean=np.mean(Data[sub,4:80,:],axis=1)

for roi in tqdm(range(ROI)):
    psd=Data[sub,4:80,roi]
    # plot(np.log(freqs),np.log(mean),alpha=.2)
    plot(freqs[4:80],psd,alpha=0.2)

plt.plot(freqs[4:80],psd_mean,'k')