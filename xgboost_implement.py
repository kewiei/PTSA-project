import numpy as np
import xgboost as xgb
import pandas as pd
import gc
import matplotlib.pyplot as plt
import time


starttime = time.time()
# read in data
mainFrame=pd.read_csv('correctmainFrame20052015.csv',parse_dates=['date'])
#model = joblib.load(r'C:\Hedge Fund Project\training\modelv1.plk')
mainFrame.set_index(['entityID','date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.dropna(axis=0,inplace=True)
mainFrame.sort_index(inplace=True)
targets=mainFrame.ix[:,-7:]
features = mainFrame.ix[:,:-7]
gc.collect()
endtime = time.time()
print("It takes {}s to load data".format(endtime-starttime))
print(mainFrame.index.dtype)

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

    dtrain = xgb.DMatrix(x_train, label=y_train)
    # specify parameters via map
    param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
    num_round = 2
    model = xgb.train(param, dtrain, num_round)

    model.fit(x_train,y_train)

    test=features[maskTest]
    
    test[np.isinf(test)]=100000000
    # pred1=model.predict_proba(test)
    dtest = xgb.DMatrix(test)
    # make prediction
    pred1 = model.predict(dtest)
    
    pred1=pd.DataFrame(pred1)
    pred1.set_index(features[maskTest].index,inplace=True)
    pred1.to_csv(r'xgboost_pred1SmallModelTest' + str(ii) + '.csv')
    gc.collect()



