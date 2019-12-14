#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from platform import python_version
print(python_version())


# In[3]:


testData=np.load(file='/beegfs/sr4376/Finance Data/CNN-LSTM Data/companyForCNN/{}ForLength{}.pkl.npz'.format(str(0),str(500)))


# In[ ]:





# In[4]:


features=testData['arr']


# In[5]:


np.shape(features)


# In[6]:


targets=pd.read_csv('/beegfs/sr4376/Finance Data/CNN-LSTM Data/CNNLSTMTargetsv2.csv',parse_dates=['date'])
targets.set_index(['entityID', 'date'],inplace=True)


# In[7]:


### load Data 
import tensorflow as tf
import keras as ks

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from random import random
from random import randint
from numpy import array
from numpy import zeros
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.callbacks import LambdaCallback
from keras import optimizers
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.utils import multi_gpu_model
import multiprocessing
from eli5.sklearn import PermutationImportance
from numba import jit


# In[8]:


np.shape(features)


# In[9]:


# function for creating a naive inception block
def inception_module(layer_in, f1, f2, f3):
# 1x1 conv
    conv1 =TimeDistributed( Conv1D(f1, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
# 3x3 conv
    conv3 = TimeDistributed(Conv1D(f2, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
# 5x5 conv
    conv5 = TimeDistributed(Conv1D(f3, kernel_size=5, padding='same', activation='relu', kernel_initializer='glorot_normal'))(layer_in)
# 3x3 max pooling
    pool = TimeDistributed(AveragePooling1D(pool_size=3, strides=1, padding='same'))(layer_in)
# concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


# In[ ]:


gc.collect()
yearsBack=np.arange(1,7)
for jj in yearsBack:
    years=np.arange(2013,2015)
    for ii in years:
        print(ii)
        lowYear=ii-jj
        maskTrain=(targets.index.get_level_values(1)>=str(lowYear)) & (targets.index.get_level_values(1)<str(ii))

        maskTest=(targets.index.get_level_values(1)>=str(ii)) & (targets.index.get_level_values(1)<str(ii+1))
        print(1)
        X_train=np.array(features[maskTrain])
        print(2)
        y_train=np.array(targets[maskTrain])
        X_train[np.isinf(X_train)]=100000000
        #X_train=np.log(X_train+1)
        y_train=y_train*1
        y_train=y_train.astype(int)
        size=377
        n_features=1
        APPENDweights=[]
        size=377

        momentum =0.9
        visible = Input(shape=(None,size,1))
        layer1 = inception_module(visible, f1=3, f2=3, f3=3)
        layer2 = TimeDistributed(BatchNormalization(momentum=momentum))(layer1)
        layer2 = TimeDistributed(Conv1D(1, kernel_size=200, activation='relu', kernel_initializer='glorot_normal'))(layer2)
        layer3 = TimeDistributed(Flatten())(layer2)
        layer4= LSTM(10, kernel_initializer='glorot_normal',bias_initializer='glorot_normal')(layer3)
        layer5 = Dense(1, activation='sigmoid')(layer4)

        model = Model(inputs=visible,outputs=layer5)
        
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 


        print(3)
        X_test=np.array(features[maskTest])
        print(4)
        X_test[np.isinf(X_test)]=100000000
        #X_test=np.log(X_test+1)
        X_test=X_test
        
        y_test=np.array(targets[maskTest])
        y_test=y_test*1
        y_test=y_test.astype(int)
        APPENDweights=[]
        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: APPENDweights.append(model.layers[-2].get_weights()[0]))

        history=model.fit(X_train, y_train, batch_size=2000, epochs=15,callbacks = [print_weights], validation_data=(X_test, y_test),verbose=1)

        pred1=model.predict(X_test,batch_size=2000)
    
        pred1=pd.DataFrame(pred1)
        pred1.set_index(targets[maskTest].index,inplace=True)
        pred1.to_csv('/beegfs/sr4376/Finance Data/InceptionLSTM/yearsBack/predRF' + str(ii) +'yearsBack' + str(jj) +'.csv')
        gc.collect()
        #perm = PermutationImportance(parallel_model, random_state=1).fit(X_test,y_test)
     
        #FI=pd.DataFrame(perm)
        #FI.to_csv('/beegfs/sr4376/Finance Data/InceptionLSTM/yearsBack/FIRF' + str(ii) +'yearsBack' + str(jj) +'.csv')
        #model=[]
        #test=[]
        #x_train=[]


# In[ ]:


parallel_model.layers[-2]


# In[ ]:


model.predict(X_test,verbose=1)


# In[ ]:




