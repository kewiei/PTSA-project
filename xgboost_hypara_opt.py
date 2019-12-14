import numpy as np
import xgboost as xgb
import pandas as pd
import gc
from sklearn.metrics import accuracy_score,r2_score
import matplotlib.pyplot as plt
import time
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

SEED = 314159265
VALID_SIZE = 0.2
TARGET = 'outcome'

starttime = time.time()
# read in data
mainFrame=pd.read_csv('/scratch/tz1264/20052007Small.csv',parse_dates=['date'])
#model = joblib.load(r'C:\Hedge Fund Project\training\modelv1.plk')
mainFrame.set_index(['entityID','date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.sort_index(inplace=True)
targets=mainFrame.iloc[:,-7:]
features = mainFrame.iloc[:,:-7]
gc.collect()
endtime = time.time()
print("It takes {}s to load data".format(endtime-starttime))

maskTrain=(mainFrame.index.get_level_values(1)>='2006') & (mainFrame.index.get_level_values(1)<'2006-10-01')

maskTest=(mainFrame.index.get_level_values(1)>='2006-10-01') & (mainFrame.index.get_level_values(1)<'2007')

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
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2

def score(params):
    print("Training with params: ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True)
    print(params)
    predictions = gbm_model.predict(dvalid,
                                    ntree_limit=gbm_model.best_iteration + 1)
    
    predictions[predictions>0.5] = 1
    predictions[predictions<=0.5] = 0
    
    score = accuracy_score(y_test, predictions)
    # TODO: Add the importance for the selected features
    print("\tScore {0}\n\n".format(score))
    # The score function should return the loss (1-score)
    # since the optimize function looks for the minimum
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}

def optimize(trials=None, 
             random_state=SEED):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 30, 1000, 10),
        'eta': hp.quniform('eta', 0.025, 1, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 20, dtype=int)),
#         'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.2, 1, 0.02),
#         'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.2, 1, 0.02),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default 
        # to the maxium number. 
#       #'nthread': 3,
        #'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'silent': 1,
        'seed': random_state
    }
    if trials is None:
        trials = Trials()
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest,
                max_evals=400, 
                trials=trials)
    return best

best_hyperparams = optimize()
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)
file = open("xgboost_best_hyperparameters.txt","w") 
file.write(str(best_hyperparams))
file.close() 
gc.collect()