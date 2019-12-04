#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:22:19 2019

@author: Amartya
"""
'''This program first finds the best hyperparameter for Logistic Regression and then uses 
those hyperparameters to train each year and predict the next year. It saves each year's predictions into a separate file.
The only two things you need to change are 1. load the right file where mainFrame is defined 
2.you'll see a for loop at the end of my code. Close to the end of the loop, in the line that saves the predictions
to a new file, set the file path of that file, but don't change the name of the file.'''

import pandas as pd
import numpy as np
import datetime as dt


import matplotlib.pyplot as plt
idx=pd.IndexSlice
from sklearn.metrics import make_scorer, r2_score,accuracy_score,precision_score
from sklearn.externals import joblib
from sklearn.grid_search import RandomizedSearchCV
import gc
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import pickle
scoring = {'AUC':'roc_auc','Accuracy':make_scorer(accuracy_score)}
mainFrame=pd.read_csv('/Volumes/Common/mainFrame20052015.csv', parse_dates=['date'])

mainFrame.set_index(['entityID','date'],inplace=True)
mainFrame.dropna(axis=0,inplace=True)
mainFrame.sort_index(inplace=True)
targets=mainFrame.iloc[:,-7:]
features = mainFrame.iloc[:,:-7]
gc.collect()

'''first we do hyperparameter training, then save the hyperparamters and then train and predict all years'''

maskTrain=(mainFrame.index.get_level_values(1)>= '2005-01-01') & (mainFrame.index.get_level_values(1)<= '2006-08-31') 

x_train=np.array(features[maskTrain])
y_train=np.array(targets['ztargetMedian5'][maskTrain])
x_train[np.isinf(x_train)]=100000000
y_train=y_train*1
y_train=y_train.astype(int)


params = {"penalty": ["l1", "l2", "elasticnet"], "dual":[True, False], "fit_intercept": [True, False], "C": np.arange(1,5,0.5), "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],"l1_ratio": np.arange(0,1.25,0.25)}

model = LogisticRegression()

random = RandomizedSearchCV(model, params)

random.fit(x_train, y_train)

best_params = random.best_params_

#save parameters
penalty = best_params["penalty"]
dual = best_params["dual"]
fit_intercept = best_params["fit_intercept"]
C= best_params["C"]
solver = best_params["solver"]
l1_ratio = best_params["l1_ratio"]

#Now we train, predict and save the predictions
years = np.arange(2005, 2015)
for ii in years:
    maskTrain=(mainFrame.index.get_level_values(1)>= str(ii)+'-01-01') & (mainFrame.index.get_level_values(1)<= str(ii)+'-12-31')
    x_train=np.array(features[maskTrain])
    y_train=np.array(targets['ztargetMedian5'][maskTrain])
    x_train[np.isinf(x_train)]=100000000
    y_train=y_train*1
    y_train=y_train.astype(int)
    model = LogisticRegression(penalty=penalty, dual=dual, fit_intercept=fit_intercept, C=C, solver=solver, l1_ratio=l1_ratio)
    model.fit(x_train, y_train)
    maskTest=(mainFrame.index.get_level_values(1) >= str(ii+1)+'-01-01') & (mainFrame.index.get_level_values(1) <= str(ii+1)+'-12-31')
    test=features[maskTest]
    test[np.isinf(test)]=100000000
    pred1=model.predict_proba(test)
    pred1=pd.DataFrame(pred1)
    pred1.set_index(features[maskTest].index,inplace=True)
    pred1.to_csv('/Users/Amartya/prediction_LogReg_'+str(ii+1)+'.csv')
    gc.collect()
    
    
    











