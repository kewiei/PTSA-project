import time
import pandas as pd
import gc
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score,r2_score
import matplotlib.pyplot as plt

starttime = time.time()
# read in data
mainFrame=pd.read_csv('/scratch/ap5891/correctmainFrame20052015.csv',parse_dates=['date'])

mainFrame.set_index(['entityID','date'],inplace=True)

mainFrame.sort_index(inplace=True)
targets=mainFrame.iloc[:,-7:]
features = mainFrame.iloc[:,:-7]
gc.collect()
endtime = time.time()
print("It takes {}s to load data".format(endtime-starttime))

def trainAndPredictOneYear(year):
    this_year = '{}'.format(year)
    next_year = '{}'.format(year+1)
    nextnext_year =  '{}'.format(year+2)
    
    print('Training on {} to predict {}'.format(this_year,next_year))
    
    maskTrain=(mainFrame.index.get_level_values(1)>=this_year) & (mainFrame.index.get_level_values(1)<next_year)
    maskTest=(mainFrame.index.get_level_values(1)>=next_year) & (mainFrame.index.get_level_values(1)<nextnext_year)

    x_train=np.array(features[maskTrain])
    y_train=np.array(targets['ztargetMedian5'][maskTrain])
    x_train[np.isinf(x_train)]=100000000
    y_train=y_train*1
    y_train=y_train.astype(int)

    x_test=np.array(features[maskTest])
    y_test = np.array(targets['ztargetMedian5'][maskTest])
    x_test[np.isinf(x_test)]=100000000
    y_test=y_test*1
    y_test=y_test.astype(int)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid =xgb.DMatrix(x_test, label=y_test)
    # specify parameters via map
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    # {'colsample_bytree': 0.26, 'eta': 0.05, 'max_depth': 15, 'n_estimators': 310.0, 'subsample': 0.68}
    param = {'booster': 'gbtree', 'colsample_bytree': 0.26, 'eta': 0.05, 'eval_metric': 'auc', 'max_depth': 15, 'n_estimators': 310.0, 'objective': 'binary:logistic', 'seed': 314159265, 'silent': 1, 'subsample': 0.68}    
    num_round = 1
    #retrain the entire model
    param['tree_method'] = 'gpu_hist'
    gbm_model = xgb.train(param, dtrain, num_round,evals=watchlist,verbose_eval=True)
    result = gbm_model.predict(dvalid)
    
    result_holder = targets.loc[maskTest,'ztargetMedian5'].copy()
    result_holder = pd.DataFrame(result_holder)
    result_holder.loc[:,'ztargetMedian5'] = result
    result_holder.to_csv('xgboost_predict{}basedon{}.csv'.format(next_year,this_year))
    

for this_year in range(2005,2015):
    trainAndPredictOneYear(this_year)
