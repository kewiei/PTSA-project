
import time
import pandas as pd
import gc
import numpy as np
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

# accurancy, precision, recall, f1, roc

starttime = time.time()
# read in data
mainFrame=pd.read_csv('truth.csv',parse_dates=['date'])

mainFrame.set_index(['entityID','date'],inplace=True)
mainFrame.sort_index(inplace=True)
mainFrame["ztargetMedian5"] = mainFrame["ztargetMedian5"].astype(int)
gc.collect()
endtime = time.time()
print("It takes {}s to load data".format(endtime-starttime))

# filePath = "logreg_result\LogRegPred{}basedon{}.csv"
# filePath = "mlp_result\mlp_predict{}basedon{}.csv"
# filePath = r"xgboost_result\xgboost_predict{}basedon{}.csv"
# filePath = "lstm_result\predRF{}yearsBack1.csv"
# filePath = "rf_result\predRF{}yearsBack1.csv"
filePath = "cnnlstm_result\predRF{}yearsBack1.csv"

pred = pd.DataFrame()
for year in range(2008,2015):
    this_year = '{}'.format(year)
    next_year = '{}'.format(year+1)
    pred_oneyear = pd.read_csv(filePath.format(this_year),parse_dates=['date'])
    # pred_oneyear.rename(columns = {"Date": 'date'},inplace=True)
    pred_oneyear.set_index(['entityID','date'],inplace=True)
    pred = pred.append(pred_oneyear)

# pred.set_index(['entityID','date'],inplace=True)
pred.sort_index(inplace=True)
# pred.rename(columns={'ztargetMedian5': '0'},inplace=True)
print(pred)
mainFrame = mainFrame.merge(pred,left_index=True,right_index=True)
mainFrame['pred'] = mainFrame['0'].apply(lambda x: 1 if x>0.5 else 0)

y_true=np.array(mainFrame["ztargetMedian5"])
y_pred=np.array(mainFrame['pred'])
y_score=np.array(mainFrame['0'])
accuracy = accuracy_score(y_true=y_true,y_pred=y_pred)
precision = precision_score(y_true=y_true,y_pred=y_pred)
recall = recall_score(y_true=y_true,y_pred=y_pred)
f1 = f1_score(y_true=y_true,y_pred=y_pred)
roc = roc_auc_score(y_true=y_true,y_score=y_score)

print("accuracy: {}".format(accuracy))
print("precision: {}".format(precision))
print("recall: {}".format(recall))
print("f1: {}".format(f1))
print("roc: {}".format(roc))

print(accuracy,precision,recall,f1,roc)
