import numpy as np
import xgboost as xgb
import pandas as pd
import gc
from sklearn.metrics import accuracy_score,r2_score
import matplotlib.pyplot as plt
import time

starttime = time.time()
# read in data
mainFrame=pd.read_csv('20052007Small.csv',parse_dates=['date'])
#model = joblib.load(r'C:\Hedge Fund Project\training\modelv1.plk')
mainFrame.set_index(['entityID','date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.dropna(axis=0,inplace=True)
mainFrame.sort_index(inplace=True)
targets=mainFrame.iloc[:,-7:]
features = mainFrame.iloc[:,:-7]
gc.collect()
endtime = time.time()
print("It takes {}s to load data".format(endtime-starttime))

maskTrain=(mainFrame.index.get_level_values(1)>=str(2005)) & (mainFrame.index.get_level_values(1)<str(2006))

maskTest=(mainFrame.index.get_level_values(1)>=str(2006)) & (mainFrame.index.get_level_values(1)<str(2007))

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

x_test=np.array(features[maskTest])
y_test = np.array(targets['ztargetMedian5'][maskTest])
x_test[np.isinf(x_test)]=100000000
y_test=y_test*1
y_test=y_test.astype(int)

# pred1=model.predict_proba(test)
dx_test = xgb.DMatrix(x_test)
# make prediction
pred1 = model.predict(dx_test)
pred1 = np.array(pred1)

accuracy = r2_score(y_test,pred1)
gc.collect()
print(accuracy)