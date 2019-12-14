import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import pandas as pd
import gc
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

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


def create_model(input_dims):
    model = keras.Sequential([
        layers.Dense(30, activation='relu',kernel_initializer='glorot_normal',bias_initializer='glorot_normal', input_shape=(input_dims,)),
        layers.BatchNormalization(momentum=0.9),
        layers.Dense(30, activation='relu',kernel_initializer='glorot_normal',bias_initializer='glorot_normal'),
        layers.BatchNormalization(momentum=0.9),
        layers.Dense(10, activation='relu',kernel_initializer='glorot_normal',bias_initializer='glorot_normal'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss='categorical_crossentropy',
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
    y_train=np.array(targets['ztargetTopDown5'][maskTrain])
    x_train[np.isinf(x_train)]=100000000
    y_train=to_categorical(y_train)

    x_test=np.array(features[maskTest])
    y_test = np.array(targets['ztargetTopDown5'][maskTest])
    x_test[np.isinf(x_test)]=100000000
    y_test=to_categorical(y_test)
    
    input_dims=x_train.shape[1]
    #retrain the entire model
    model = create_model(input_dims)
    model.fit(x_train, y_train, batch_size=500, epochs=200, 
          validation_split=0.1, verbose=1)
    
    result = model.predict(x_test)
    result_holder = targets.loc[maskTest,'ztargetTopDown5'].copy()
    result_holder = pd.DataFrame(result_holder)
    result_holder.rename(columns={"ztargetTopDown5":"0"},inplace=True)
    result_holder["1"] = None
    result_holder["2"] = None
    result_holder.loc[:,['0','1','2']] = result
    result_holder.to_csv('mlp_multiclass_predict{}basedon{}.csv'.format(next_year,this_year))    

for this_year in range(2005,2015):
    trainAndPredictOneYear(this_year)