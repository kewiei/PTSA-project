'''Code to transform the entire excel file. Please change the read file path and the new path'''
import pandas as pd
import numpy as np
row_list = []
x=pd.read_csv('/Volumes/Common/returnData.csv')
stocks = x.entityID.unique()
for i in stocks:
    indices = x[x.entityID == i].index
    for j in indices:
        if j+240 in indices:
            dict1 = dict(zip(np.arange(1,241).tolist(), x['return1'][j:j+240].tolist()))
            dict1['Date'] = x['date'][j+240]
            dict1['entityID'] = i
            dict1['ztargetMedian5'] = str(x['ztargetMedian5'][j+240])
            row_list.append(dict1)
        else:
            break
df = pd.DataFrame(row_list, columns = ['entityID','Date'] + np.arange(1,241).tolist() + ['ztargetMedian5'])
df.to_csv('/Volumes/Common/TransformedData-LSTM')       