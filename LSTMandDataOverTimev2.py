#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### load Data 
import tensorflow as tf
import tensorflow.keras as ks

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

from random import random
from random import randint
from numpy import array
from numpy import zeros
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import optimizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import multi_gpu_model
import multiprocessing


# In[ ]:


import multiprocessing
#import dask.dataframe as dk
import pandas as pd
import numpy as np
import datetime as dt

#import matplotlib.pyplot as plt
idx=pd.IndexSlice
from sklearn.metrics import make_scorer, r2_score,accuracy_score,precision_score
from sklearn.externals import joblib
import os
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from tqdm import tqdm


# In[ ]:


multiprocessing.cpu_count()


# In[ ]:


mainFrame=pd.read_csv('/beegfs/sr4376/Finance Data/CNN-LSTM Data/TransformedData-LSTM.csv',parse_dates=["Date"])

#model = joblib.load(r'C:\Hedge Fund Project\training\modelv1.plk')


# In[ ]:


mainFrame.columns


# In[ ]:


mainFrame.drop(columns=['Unnamed: 0'],inplace=True)
mainFrame.set_index(['entityID','Date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.dropna(axis=0,inplace=True)
mainFrame.sort_index(inplace=True)
#mainFrame=mainFrame[~(mainFrame.index.get_level_values(1)>'2012')]

targets=mainFrame['ztargetMedian5']
features = mainFrame.ix[:,:-1]


# In[ ]:


features


# In[ ]:


X_train=np.array(features[mainFrame.index.get_level_values(1)<str(2012)])
y_train=np.array(targets[mainFrame.index.get_level_values(1)<str(2012)])
X_train=np.reshape(X_train,(np.shape(X_train)[0],np.shape(X_train)[1],1))

X_test=np.array(features[mainFrame.index.get_level_values(1)>=str(2012)])
y_test=np.array(targets[mainFrame.index.get_level_values(1)>=str(2012)])
X_test=np.reshape(X_test,(np.shape(X_test)[0],np.shape(X_test)[1],1))


# In[ ]:

gc.collect()
yearsBack=np.arange(1,7)
for jj in yearsBack:
    years=np.arange(2008,2015)
    for ii in years:
        print(ii)
        lowYear=ii-jj
        maskTrain=(features.index.get_level_values(1)>=str(lowYear)) & (features.index.get_level_values(1)<str(ii))

        maskTest=(features.index.get_level_values(1)>=str(ii)) & (features.index.get_level_values(1)<str(ii+1))

        X_train=np.array(features[maskTrain])
        y_train=np.array(targets[maskTrain])
        X_train[np.isinf(X_train)]=100000000
        X_train=np.reshape(X_train,(np.shape(X_train)[0],np.shape(X_train)[1],1))
        X_train=np.log(X_train+1)
        y_train=y_train*1
        y_train=y_train.astype(int)
        length=240
        n_features=1
        model = Sequential()
        model.add(LSTM(30, input_shape=(240,1)))
        model.add(Dense(10))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])


        X_test=np.array(features[maskTest])
    
        X_test[np.isinf(X_test)]=100000000
        X_test=np.reshape(X_test,(np.shape(X_test)[0],np.shape(X_test)[1],1))
        X_test=np.log(X_test+1)
        X_test=X_test
        
        y_test=np.array(targets[maskTest])
        y_test=y_test*1
        y_test=y_test.astype(int)
        APPENDweights=[]
        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: APPENDweights.append(model.layers[-2].get_weights()[0]))

        model.fit(X_train, y_train, batch_size=2000, epochs=50,callbacks = [print_weights], validation_data=(X_test, y_test),verbose=1)
        print(1)
        pred1=model.predict_proba(X_test, batch_size=2000)
        print(2)
        pred1=pd.DataFrame(pred1)
        print(3)
        pred1.set_index(targets[maskTest].index,inplace=True)
        pred1.to_csv('/beegfs/sr4376/Finance Data/LSTM/yearsBack/predRF' + str(ii) +'yearsBack' + str(jj) +'.csv')
        gc.collect()
        print(4)
        #perm = PermutationImportance(model, random_state=1).fit(X_test,y_test)
        #eli5.show_weights(perm, feature_names = X.columns.tolist())
        #FI=pd.DataFrame(perm)
        #FI.to_csv('/beegfs/sr4376/Finance Data/LSTM/yearsBack/FIRF' + str(ii) +'yearsBack' + str(jj) +'.csv')
        #model=[]
        #test=[]
        #x_train=[]


# In[ ]:


pred1=model.predict_proba(X_test)


# In[ ]:




