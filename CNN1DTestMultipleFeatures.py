#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:02:06 2023

@author: sflores
"""

import numpy as np
import matplotlib.pyplot as plt
from Fun3 import *
from FunClassifier import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


#%%
# ==============================================================================

def ACP (DataFrame,verbose,scatterMatrix,nPca,labels):
    if verbose:
        
        print('----------------------')
        print('Media de cada variable')
        print('----------------------')
        print(DataFrame.mean(axis=0))
        
        
        print('-------------------------')
        print('Varianza de cada variable')
        print('-------------------------')
        print(DataFrame.var(axis=0))
    
    #Etiqueta de componentes a calcular
    # ==========================================================================
    Etiquetas=[]
    for i in range(len(DataFrame.keys())):
        Etiquetas.append('PC'+str(i+1))
    
    # Entrenamiento modelo PCA con escalado de los datos
    # ==============================================================================
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(DataFrame)
    
    # Se extrae el modelo entrenado del pipeline
    modelo_pca = pca_pipe.named_steps['pca']
    
    dfPca=pd.DataFrame(
        data    = modelo_pca.components_,
        columns = DataFrame.columns,
        index   = Etiquetas
    )
    if verbose:
        print(dfPca)
    
    # Heatmap componentes
    # ==============================================================================
    
    componentes = modelo_pca.components_
    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
        plt.imshow(componentes.T, cmap='viridis', aspect='auto')
        plt.yticks(range(len(DataFrame.columns)), DataFrame.columns)
        plt.xticks(range(len(DataFrame.columns)), np.arange(modelo_pca.n_components_) + 1)
        plt.grid(False)
        plt.colorbar();
    
    
    # Porcentaje de varianza explicada por cada componente
    # ==============================================================================
    if verbose:
        print('----------------------------------------------------')
        print('Porcentaje de varianza explicada por cada componente')
        print('----------------------------------------------------')
        print(modelo_pca.explained_variance_ratio_)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.bar(
            x      = np.arange(modelo_pca.n_components_) + 1,
            height = modelo_pca.explained_variance_ratio_
        )
    
        for x, y in zip(np.arange(len(DataFrame.columns)) + 1, modelo_pca.explained_variance_ratio_):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
    
        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_ylim(0, 1.1)
        ax.set_title('Porcentaje de varianza explicada por cada componente')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza explicada');
    
    # Porcentaje de varianza explicada acumulada
    # ==============================================================================
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    if verbose:
        print('------------------------------------------')
        print('Porcentaje de varianza explicada acumulada')
        print('------------------------------------------')
        print(prop_varianza_acum)
    
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        ax.plot(
            np.arange(len(DataFrame.columns)) + 1,
            prop_varianza_acum,
            marker = 'o'
        )
    
        for x, y in zip(np.arange(len(DataFrame.columns)) + 1, prop_varianza_acum):
            label = round(y, 2)
            ax.annotate(
                label,
                (x,y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center'
            )
            
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
        ax.set_title('Porcentaje de varianza explicada acumulada')
        ax.set_xlabel('Componente principal')
        ax.set_ylabel('Por. varianza acumulada');
    
    proyecciones = pca_pipe.transform(X=DataFrame)
    proyecciones = pd.DataFrame(
        proyecciones,
        columns = Etiquetas,
        index   = DataFrame.index
    )
    pro2use=proyecciones.iloc[:,:nPca]
    pro2use['Cohort']=labels
    if verbose:
        print(proyecciones.head())
    
    if scatterMatrix:
        g=sns.pairplot(pro2use, hue="Cohort",corner=True)
        g.map_lower(sns.kdeplot, levels=6, color=".2")
    
    return proyecciones,prop_varianza_acum

#%%
Data,Aper = GetStuff('alpha', Windows=99, 
                      sampleSize=250,seed=2, 
                      plot=False, in_between=[1,40],
                      max_n_peaks=6, fit='fixed')

#%%
freqs=Data.freqs
PSD=np.array(Aper.whitened)
plt.figure()
meanA=np.mean(PSD[:int(PSD.shape[0]/2)],axis=0)
meanB=np.mean(PSD[int(PSD.shape[0]/2):],axis=0)
stdA=np.std(PSD[:int(PSD.shape[0]/2)],axis=0)
stdB=np.std(PSD[int(PSD.shape[0]/2):],axis=0)
# plt.plot(np.log(freqs),np.log(meanA),'r')
# plt.plot(np.log(freqs),np.log(meanB),'g')
print(np.sqrt(PSD.shape[0]/2))
coefIntervalA=(1.96*(stdA)/np.sqrt(PSD.shape[0]/2))
coefIntervalB=(1.96*(stdB)/np.sqrt(PSD.shape[0]/2))
# plt.fill_between(np.log(freqs),np.log(meanA+coefIntervalA),np.log(meanA-coefIntervalA),alpha=.5,color='r')
# plt.fill_between(np.log(freqs),np.log(meanB+coefIntervalB),np.log(meanB-coefIntervalB),alpha=.5,color='g')
plt.plot(freqs,meanA,'r')
plt.plot(freqs,meanB,'g')
# plt.fill_between(freqs,meanA+coefIntervalA,meanA-coefIntervalA,alpha=.5,color='r')
# plt.fill_between(freqs,meanB+coefIntervalB,meanB-coefIntervalB,alpha=.5,color='g')
plt.fill_between(freqs,meanA+stdA,meanA-stdA,alpha=.5,color='r')
plt.fill_between(freqs,meanB+stdB,meanB-stdB,alpha=.5,color='g')


plt.title('Band= '+Generate_Data.band+', Mean A vs B groups, all widows, all sujects. Freqs between'+str(Data.in_between))
plt.show()

#%%
labels=Aper.periodic['Cohort'].to_numpy()
matrix=Aper.whitened.to_numpy()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_trained_samples=scaler.fit_transform(matrix)
# PSD = matrix.reshape(matrix.shape[0], matrix.shape[1], 1)
PSD = scaled_trained_samples.reshape(scaled_trained_samples.shape[0], scaled_trained_samples.shape[1], 1)

x_train, x_test, y_train, y_test = train_test_split(PSD,
                                                    labels,
                                                    test_size=.3)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
tf.keras.backend.clear_session()
model = Sequential ([
    Conv1D(128,25,activation='relu',input_shape=(x_train.shape[1],1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, strides=1),
    Conv1D(256,15,activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, strides=1),
    Conv1D(128,5,activation='relu'),
    BatchNormalization(),
    # GlobalMaxPooling1D(),
    Flatten(),
    # Dense(512,activation='tanh'),
    # Dense(128,activation='relu'),
    # Dropout(.1),
    Dense(32,activation='tanh'),
    # Dropout(.1),
    Dense(64,activation='relu'),
    # Dropout(.3),
    Dense(2,activation='softmax')
    
    ])

print(model.summary())
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='SparseCategoricalCrossentropy',
              metrics=[Accuracy(),Precision(),Recall()])

history=model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.1,
    verbose=1)

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=64)
print(results)
# return results

#%%

labels=Aper.periodic['Cohort'].to_numpy()
matrix=Aper.whitened
proyecciones,prop_varianza_acum=ACP(matrix,True,True,5,labels)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_trained_samples=scaler.fit_transform(matrix.to_numpy())
X=proyecciones.to_numpy()[:,:20]
y=labels
x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                      test_size=.3)
C = .1  # SVM regularization parameter
clf = svm.SVC(C=C,kernel='rbf').fit(x_train, y_train)
    

title ="SVC with linear kernel"
    

# Set-up 2x2 grid for plotting.
# fig, ax = plt.subplots(1, 1)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
y_pred=clf.predict(x_test)
acc=accuracy_score(y_test, y_pred)*100
print(acc)
# disp = DecisionBoundaryDisplay.from_estimator(
#     clf,
#     X,
#     response_method="predict",
#     cmap=plt.cm.coolwarm,
#     alpha=0.8,
#     ax=ax,
#     xlabel='Alpha_Power',
#     ylabel='Beta_Power',
# )
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title+', acc= '+str((accuracy_score(y_test, y_pred)*100)))

