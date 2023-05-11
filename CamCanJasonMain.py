#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:03:35 2023

@author: isaac
"""
import os
import pandas as pd
import numpy as np
current_path = os.getcwd()
parentPath=os.path.abspath(os.path.join(current_path,os.pardir))
path2Data=ParentPath+'/Stability-project_db/CAMCAN_Jason_PrePro/'
mainDir=np.sort(os.listdir(Path2Data))
demographics=pd.read_csv(path2Data+mainDir[0],header=None)