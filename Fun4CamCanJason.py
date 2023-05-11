#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:58:43 2023

@author: isaac
"""

import numpy as np

def myReshape(array):
    [x,y]=array.shape
    newarray=np.zeros((68,606,300))
    for i,j in enumerate(np.arange(0,y,300)):
        newarray[i,:,:]=array[:,j:j+300]
        
    return newarray