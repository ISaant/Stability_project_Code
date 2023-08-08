#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:41:11 2023

@author: isaac
"""

# Perceptron on PCA Data per ROI

Sub,PSD,ROI=restStateCropped.shape
nPCA=10
restStatePCA=np.zeros((Sub,nPCA,ROI))
RrestStateCropped= Relativize(restStateCropped)
for roi in range(ROI):
    pca_df, pca2use, prop_varianza_acum= myPCA(np.log(RrestStateCropped[:,:,roi]),False,nPCA)
    plt.plot(prop_varianza_acum[:10])
    restStatePCA[:,:,roi]=np.array(pca2use)
    
restStatePCA=RestoreShape(restStatePCA)

#%%

def TestAlgorithmsRegression_psdPCA(restStatePCA,Age,Catell):
    AAge=copy.copy(Age)
    CCatell=copy.copy(Catell)
    CrestStatePCA=copy.copy(restStatePCA)
    # Delete outliers from target drop same subject, we will use all regions =========

    # AAge=np.delete(AAge, idxOutliers)
    # CCatell=np.delete(CCatell, idxOutliers)
    # CrestStatePCA=np.delete(CrestStatePCA, idxOutliers,axis=0)  
    
    #

    def GenerateRegressions (Data,labels):
        Reg=[]
        # Delete nan from target drop same subject, we will use all regions =========

        Data,labels=RemoveNan(Data, labels)
        DataScaled=Scale(Data)
        
        for j in tqdm(range(100)):
        # Data,labels=RemoveNan(np.log(restStateOriginal), label)
            #x_train, x_test, y_train,y_test =Split(DataScaled,labels,.2)
            #x_train2, x_test2, y_train2,y_test2=Split(Data2[:,:50],labels2,.2)
            x_train, x_test, y_train,y_test,_,_=Split(DataScaled,labels,.5)
            # x_train, x_test, y_train,y_test,_,_=Split(Data,labels,.2)

        # Lasso
            model = Lasso(alpha=.2)
            model.fit(x_train, y_train)
            pred_Lasso=model.predict(x_test)
            LassoPred=plotPredictionsReg(pred_Lasso,y_test,False)
    
    
        # Perceptron Regression
            Input0=tf.keras.Input(shape=(x_train.shape[1],), )
            modelNN=Perceptron_PCA (Input0,False)
            trainModel(modelNN,x_train,y_train,100,True)
            # predNN=evaluateRegModel(model,x_test,y_test)
            predNN = modelNN.predict(x_test).flatten()
            NNPred=plotPredictionsReg(predNN,y_test,False)
            
        # Random Forest
            model=RandomForestRegressor(n_estimators=20)
            model.fit(x_train, y_train)
            pred_Rf=model.predict(x_test)
            RFPred=plotPredictionsReg(pred_Rf,y_test,False)
            plt.close('all')
            Reg.append([LassoPred, NNPred, RFPred ])
        
        return np.array(Reg)
        
    
    AgeReg=GenerateRegressions(CrestStatePCA,AAge)
    CatellReg=GenerateRegressions(CrestStatePCA,CCatell)
    return AgeReg, CatellReg
AgeReg, CatellReg=TestAlgorithmsRegression_psdPCA(restStatePCA,Age,Catell)

#%%
def WeDontLikeNaNs(Reg):
    idx=np.argwhere(np.isnan(Reg[:,1]))
    Reg=np.delete(Reg, idx,axis=0)
    idx=np.argwhere(Reg[:,1]<.2)
    Reg=np.delete(Reg, idx,axis=0)
    return Reg

AAgeReg=np.ndarray.flatten(WeDontLikeNaNs(AgeReg))
CCatellReg=np.ndarray.flatten(WeDontLikeNaNs(CatellReg))

#%%
DfpsdPCAReg=pd.DataFrame({'Corr':np.concatenate((AAgeReg,CCatellReg)),
                        'Algorithm':np.array(['Lasso','NN','RF']*int((len(AAgeReg)/3)+len(CCatellReg)/3)),
                        'Test':np.array(['Age']*len(AAgeReg)+['Catell']*len(CCatellReg))})        

sns.set_context("poster", font_scale = .8, rc={"grid.linewidth": 5})
sns.boxplot(DfpsdPCAReg,x='Test',y='Corr',hue='Algorithm',palette='viridis').set_title('Predictibility per Algorithm')
