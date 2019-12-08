import pandas as pd
import numpy as np
idx=pd.IndexSlice
mainFrame=pd.read_csv('/scratch/ap5891/correctmainFrame20052015.csv',parse_dates=['date'])

mainFrame.set_index(['entityID','date'],inplace=True)
#mainFrame=mainFrame20052018
mainFrame.sort_index(inplace=True)
features = mainFrame.ix[:,:-7]
mainFrame=[]
returns=features[['return1','return2']]
features=[]
pred1 = None

for year in range(2005,2015):
    this_year = '{}'.format(year)
    next_year = '{}'.format(year+1)
    tmpPred=pd.read_csv('mlp_predict{}basedon{}.csv'.format(next_year,this_year),parse_dates=['date'])
    tmpPred.set_index(['entityID','date'],inplace=True)
    tmpPred.sort_index(inplace=True)
    if pred1 is None:
        pred1=tmpPred
    else:
        pred1=pred1.append(tmpPred)
                
tradingSetsLong={}
tradingSetsShort={}
portfolios={0:[],1:[],2:[],3:[],4:[]}
portfolios[0].append(1000000)
portfolios[1].append(1000000)
portfolios[2].append(1000000)
portfolios[3].append(1000000)   
portfolios[4].append(1000000)
portfolioValue=[]
portfolioValue.append(1000000)


proba=pred1
proba.sort_index(inplace=True)
proba.rename(columns={'0':'Short','1':'Long'},inplace=True)
dateVectorPnL = pd.to_datetime(proba.index.get_level_values('date').unique().sort_values())
counter=0
stockNumber=50
for ii in dateVectorPnL[:-1]:
    print(ii)
    idxDate=dateVectorPnL.get_loc(ii)
    tomorrow=dateVectorPnL[idxDate+1]
    if counter>4:
        counter=0
    longStocks=pd.DataFrame(proba.xs(ii,axis=0,level=1,drop_level=False)['ztargetMedian5'].nlargest(stockNumber))
    shortStocks=pd.DataFrame(proba.xs(ii,axis=0,level=1,drop_level=False)['ztargetMedian5'].nsmallest(stockNumber))
    longEntity=longStocks.index.get_level_values('entityID')
    shortEntity=shortStocks.index.get_level_values('entityID')
    tradingSetsLong[counter]=longEntity
    tradingSetsShort[counter]=shortEntity    

    for ll in np.arange(0,5):
        if len(tradingSetsLong)>=ll+1:
            longPnL = np.sum(returns.loc[idx[tradingSetsLong[ll],:]]['return1'].xs(tomorrow,axis=0,level=1,drop_level=False)*portfolios[ll][-1]/stockNumber)
            shortPnL = np.sum(returns.loc[idx[tradingSetsShort[ll],:]]['return1'].xs(tomorrow,axis=0,level=1,drop_level=False)*portfolios[ll][-1]/stockNumber)
            portfolios[ll].append(portfolios[ll][-1]+longPnL-shortPnL)
        else:
            portfolios[ll].append(portfolios[ll][-1])
    print(portfolios[0][-1])
    print(portfolios[1][-1])
    print(portfolios[2][-1])
    print(portfolios[3][-1])
    print(portfolios[4][-1])
    counter = counter  +1  
portfolios=pd.DataFrame(portfolios)
portfolios.set_index(dateVectorPnL,inplace=True)
portfolios.to_csv('MLPReturnAnalyse.csv')