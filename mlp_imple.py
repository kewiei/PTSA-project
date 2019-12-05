import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import pandas as pd
import gc
import numpy as np
import os

model_path = 'mlp_imple.h5'

starttime = time.time()
# read in data
mainFrame=pd.read_csv('correctmainFrame20052015.csv',parse_dates=['date'])
#model = joblib.load(r'C:\Hedge Fund Project\training\modelv1.plk')
mainFrame.set_index(['entityID','date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.sort_index(inplace=True)
targets=mainFrame.iloc[:,-7:]
features = mainFrame.iloc[:,:-7]
gc.collect()
endtime = time.time()
print("It takes {}s to load data".format(endtime-starttime))

# Parameters
def create_model(input_dims):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dims,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.binary_crossentropy,
                 metrics=['accuracy'])
    return model

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

    input_dims=x_train.shape[1]
    #retrain the entire model
    model = create_model(input_dims)
    model.fit(x_train, y_train, batch_size=50, epochs=50, 
          validation_split=0.1, verbose=1)
    
    result = model.predict(x_test)
    result_holder = targets.loc[maskTest,'ztargetMedian5'].copy()
    result_holder = pd.DataFrame(result_holder)
    result_holder.loc[:,'ztargetMedian5'] = result
    result_holder.to_csv('mlp_predict{}basedon{}.csv'.format(this_year,next_year))
    

for this_year in range(2005,2014):
    trainAndPredictOneYear(this_year)