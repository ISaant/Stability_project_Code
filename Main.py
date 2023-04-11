#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:19:49 2023

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from Fun2 import *


APer_Alpha_1peak = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=False, in_between=[1,50],
                      max_n_peaks=1, fit='fixed')

coeffs=APer_Alpha_1peak.get_Parameters()
# APer_Alpha_6peak = GetStuff('alpha', Windows=99, 
#                       sampleSize=250,seed=2, 
#                       plot=False, in_between=[1,250],
#                       max_n_peaks=6, fit='knee')

# mfcc=APer_Alpha_6peak.mfcc
# filter_banks=APer_Alpha_6peak.filter_banks
# Data_Beta,APer_Beta = GetStuff('beta', Windows=99, 
#                                   sampleSize=250,seed=2, 
#                                   plot=False, in_between=[1,250],
#                                   max_n_peaks=1, fit='knee')