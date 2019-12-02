# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:31:05 2019

@author: sr4376
"""

import pandas as pd
import numpy as np
import datetime as dt


import matplotlib.pyplot as plt
idx=pd.IndexSlice
from sklearn.metrics import make_scorer, r2_score,accuracy_score,precision_score
from sklearn.externals import joblib
from numba import jit
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

import pickle
scoring = {'AUC':'roc_auc','Accuracy':make_scorer(accuracy_score)}

mainFrame=pd.read_csv(r'C:\Hedge Fund Project\Generate Features\data\mainFrame\mainFrame20052018.csv',parse_dates=['date'])
#model = joblib.load(r'C:\Hedge Fund Project\training\modelv1.plk')

mainFrame.set_index(['entityID','date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.dropna(axis=0,inplace=True)
mainFrame.sort_index(inplace=True)
targets=mainFrame.ix[:,-7:]
features = mainFrame.ix[:,:-7]
gc.collect()

years=np.arange(2011,2012)
for ii in years:
    print(ii)
    lowYear=ii-3
    maskTrain=(mainFrame.index.get_level_values(1)>=str(lowYear)) & (mainFrame.index.get_level_values(1)<str(ii))

    maskTest=(mainFrame.index.get_level_values(1)>=str(ii)) & (mainFrame.index.get_level_values(1)<str(ii+1))

    x_train=np.array(features[maskTrain])
    y_train=np.array(targets['ztargetMedian5'][maskTrain])
    x_train[np.isinf(x_train)]=100000000
    y_train=y_train*1
    y_train=y_train.astype(int)
    model = RandomForestClassifier(n_estimators=200,max_depth=20,n_jobs=24,verbose=3)

    model.fit(x_train,y_train)

    test=features[maskTest]
    
    test[np.isinf(test)]=100000000
    pred1=model.predict_proba(test)
    
    pred1=pd.DataFrame(pred1)
    pred1.set_index(features[maskTest].index,inplace=True)
    pred1.to_csv(r'C:\Hedge Fund Project\prediction\prediciton1\pred1SmallModelTest' + str(ii) + '.csv')
    gc.collect()

    

y_test=np.array(targets['ztargetMedian5'][maskTest])

y_test=y_test*1
y_test=y_test.astype(int)
predictions = []
for tree in model.estimators_:
    predictions.append(tree.predict_proba(X_test)[None, :])
    
predictions = np.vstack(predictions)

cum_mean = np.cumsum(predictions, axis=0)/np.arange(1, predictions.shape[0] + 1)[:, None, None]

scores = []
for pred in cum_mean:
    scores.append(accuracy_score(y_test, np.argmax(pred, axis=1)))
    
plt.figure(figsize=(10, 6))
plt.plot(scores, linewidth=3)
plt.xlabel('num_trees')
plt.ylabel('accuracy');
