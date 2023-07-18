#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:00:03 2023

@author: sflores
"""

import tqdm
import pandas as pd
from FunClassifier4CamCanJason import *
from Fun4CamCanJason import *
def TestAlgorithmsRegression_Anatomical(CorticalThickness,Age,Catell):
    AAge=copy.copy(Age)
    CCatell=copy.copy(Catell)
    CCorticalThickness=copy.copy(CorticalThickness.to_numpy())
    idxOutliers=np.array([119,491])
    # Delete outliers from target drop same subject, we will use all regions =========

    AAge=np.delete(AAge, idxOutliers)
    CCatell=np.delete(CCatell, idxOutliers)
    CCorticalThickness=np.delete(CCorticalThickness, idxOutliers,axis=0)  
    
    #

    def GenerateRegressions (Data,labels):
        Reg=[]
        # Delete nan from target drop same subject, we will use all regions =========

        Data,labels=RemoveNan(Data, labels)
        # DataScaled=Scale(Data)
        
        for j in range(100):
        # Data,labels=RemoveNan(np.log(restStateOriginal), label)
            #x_train, x_test, y_train,y_test =Split(DataScaled,labels,.2)
            #x_train2, x_test2, y_train2,y_test2=Split(Data2[:,:50],labels2,.2)
            x_train, x_test, y_train,y_test,_,_=Split(Data,labels,.2)

        # Lasso
            model = Lasso(alpha=.2)
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            LassoPred=plotPredictionsReg(pred_Lasso,y_test)
    
    
        # Perceptron Regression
            Input0=tf.keras.Input(shape=(x_train.shape[1],), )
            modelNN=Perceptron (Input0,False)
            trainModel(modelNN,x_train,y_train,500,True)
            # predNN=evaluateRegModel(model,x_test,y_test)
            predNN = modelNN.predict(x_test).flatten()
            NNPred=plotPredictionsReg(predNN,y_test)
            
        # Random Forest
            model=RandomForestRegressor(n_estimators=20)
            model.fit(x_train, y_train)
            pred_Rf=model.predict(x_test)
            RFPred=plotPredictionsReg(pred_Rf,y_test)
            plt.close('all')
            Reg.append([LassoPred, NNPred, RFPred ])
        
        return np.array(Reg)
        
    
    AgeReg=GenerateRegressions(CCorticalThickness,AAge)
    CatellReg=GenerateRegressions(CCorticalThickness,CCatell)
    return AgeReg, CatellReg
AgeReg, CatellReg=TestAlgorithmsRegression_Anatomical(CorticalThickness,Age,Catell)
        # with open(current_path+'/Pickle/AgePredictions2.pickle', 'wb') as f:
        #     pickle.dump(AGE, f)
        # with open(current_path+'/Pickle/CatellPredictions2.pickle', 'wb') as f:
        #     pickle.dump(CATELL, f)
        # with open(current_path+'/Pickle/AcerPredictions2.pickle', 'wb') as f:
        #     pickle.dump(ACER, f)