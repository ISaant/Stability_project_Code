#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:02:20 2023

@author: sflores
"""
import tqdm
import pandas as pd
from FunClassifier4CamCanJason import *
from Fun4CamCanJason import *
def TestAlgorithmsRegression(restStateDir,path2Data,mainDir,columns,):
    Targets=['Catell','Age','Acer']
    ACER=[]
    AGE=[]
    CATELL=[]
    # Input0=tf.keras.Input(shape=(70,), )
    # modelNN=Perceptron (Input0,False)
    for i in tqdm(range(10)):
        TimeWindows=restStateDir[0:i+1]
        if i == 0:
            print(i)
            TimeWindows=[restStateDir[0]]
        for e,file in enumerate(TimeWindows):
            matrix=pd.read_csv(path2Data+mainDir[1]+'/'+file,header=None)
            if e == 0:
                # print (e)
                restStateOriginal=matrix
                continue
            restStateOriginal+=matrix
        restStateOriginal=restStateOriginal
        restStateOriginal/=(e+1)
        restState = myReshape(restStateOriginal.to_numpy()) #reshape into [Subjects,PSD,ROI]
        restStateCropped = restState[:,columns,:] # Select the band-width of interest

    # Plot global mean and mean per ROI
    # psdPlot(freqs[columns], restStateCropped)


        nPca=100
        # pca_df,pro2use,prop_varianza_acum=myPCA (np.log(restStateOriginal),True, nPca)
        W = NNMatFac(restStateOriginal.to_numpy(),nPca)
        
    # Delete nan from target drop same subject, we will use all regions =========
        for target in Targets:
            label=eval(target)
            exec(target+'Reg=[]')
            print (target)
            # Data,labels=RemoveNan(np.log(restStateOriginal), label)
            # DataScaled=Scale(Data)
            # Data3,labels2=RemoveNan(pca_df, label)
            # DataScaled3=Scale(Data3)
            Data,labels=RemoveNan(W, label)
            DataScaled=Scale(Data)
            for j in range(100):
            # Data,labels=RemoveNan(np.log(restStateOriginal), label)
                #x_train, x_test, y_train,y_test =Split(DataScaled,labels,.2)
                #x_train2, x_test2, y_train2,y_test2=Split(Data2[:,:50],labels2,.2)
                x_train, x_test, y_train,y_test=Split(DataScaled,labels,.2)

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
                exec(target+'Reg.append([LassoPred, NNPred, RFPred ])')
                
            A=np.array(eval(target+'Reg'))
            # exec(target.upper()+'.append([[np.mean(A[:,0]),np.std(A[:,0])], [np.mean(A[:,1]),np.std(A[:,1])] ,[np.mean(A[:,2]),np.std(A[:,2])]])')
            exec(target.upper()+'.append(A)')
        # with open(current_path+'/Pickle/AgePredictions2.pickle', 'wb') as f:
        #     pickle.dump(AGE, f)
        # with open(current_path+'/Pickle/CatellPredictions2.pickle', 'wb') as f:
        #     pickle.dump(CATELL, f)
        # with open(current_path+'/Pickle/AcerPredictions2.pickle', 'wb') as f:
        #     pickle.dump(ACER, f)