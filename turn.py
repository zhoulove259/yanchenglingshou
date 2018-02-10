import pandas as pd
import numpy as np
import datetime
import time
import xgboost as xgb


dataTrain=pd.read_csv('data/dataTrain.csv')

def splitTR(x):
    first=0
    second=0
    if ':' in x:
        first,second=x.split(';')
    return (first+second)/2


def get_price_level(x):
    if x=='5WL':
        return 0
    if x=='5-8W':
        return 1
    if x=='8-10W':
        return 2
    if x=='10-15W':
        return 3
    if x=='15-20W':
        return 4
    if x=='20-25W':
        return 5
    if x=='25-35W':
        return 6
    if x=='35-50W':
        return 7
    if x=='50-75W':
        return 8


def get_power(x):
    arr=x.split('/')
    if len(arr)==1:
        return arr
    return (int(arr[0])+int(arr[1]))/2


def get_engine_torque(x):
    arr=x.split('/')
    if len(arr)==1:
        return arr
    return (int(arr[0])+int(arr[1]))/2

def get_rated_passenger(x):
    arr=x.split('-')
    if len(arr)==1:
        return arr
    return (int(arr[0])+int(arr[1]))/2


def data_turn(dataTrain):
    # turn level_id
    dataTrain.level_id.replace('-', np.nan, inplace=True)
    dataTrain.level_id = dataTrain.level_id.astype('float')

    # turn TR
    dataTrain.TR = dataTrain.TR.apply(lambda x: splitTR(x) if ';' in x else x)
    dataTrain.TR = dataTrain.TR.astype('float')

    # turn gearbox_type
    gearbox_type_dummies = pd.get_dummies(dataTrain.gearbox_type)
    gearbox_type_dummies.columns = ['gearbox_type' + str(i + 1) for i in range(gearbox_type_dummies.shape[1])]
    dataTrain = pd.concat([dataTrain, gearbox_type_dummies], axis=1)
    dataTrain.drop(['gearbox_type'], axis=1, inplace=True)

    # turn if_charging
    if_charging_dummies = pd.get_dummies(dataTrain.if_charging)
    if_charging_dummies.columns = ['if_charging' + str(i + 1) for i in range(if_charging_dummies.shape[1])]
    dataTrain = pd.concat([dataTrain, if_charging_dummies], axis=1)
    dataTrain.drop(['if_charging'], axis=1, inplace=True)

    # turn price_level
    dataTrain.price_level = dataTrain.price_level.apply(get_price_level)
    dataTrain.price_level = dataTrain.price_level.astype('float')

    # turn price
    dataTrain.price.replace('-', np.nan, inplace=True)
    dataTrain.price = dataTrain.price.astype('float')

    # turn fuel_type_id
    dataTrain.fuel_type_id.replace('-', np.nan, inplace=True)
    dataTrain.fuel_type_id = dataTrain.fuel_type_id.astype('float')

    # turn power
    dataTrain.power = dataTrain.power.astype('str')
    dataTrain.power = dataTrain.power.apply(lambda x: get_power(x) if '/' in x else x)
    dataTrain.power = dataTrain.power.astype('float')

    # turn engine_torque
    dataTrain.engine_torque = dataTrain.engine_torque.astype('str')
    dataTrain.engine_torque = dataTrain.engine_torque.apply(lambda x: get_engine_torque(x) if '/' in x else x)
    dataTrain.engine_torque.replace('-', np.nan, inplace=True)
    dataTrain.engine_torque = dataTrain.engine_torque.astype('float')

    # turn rated_passenger
    dataTrain.rated_passenger = dataTrain.rated_passenger.astype('str')
    dataTrain.rated_passenger = dataTrain.rated_passenger.apply(lambda x: get_rated_passenger(x) if '-' in x else x)
    dataTrain.rated_passenger = dataTrain.rated_passenger.astype('float')

    # turn time
    dataTrain.sale_date=dataTrain.sale_date*100+1
    dataTrain.sale_date=dataTrain.sale_date.astype('str')
    dataTrain.sale_date=dataTrain.sale_date.apply(lambda x: datetime.datetime(int(x[0:4]),int(x[4:6]),int(x[6:8])))
    dataTrain['year']=dataTrain.sale_date.apply(lambda x:x.year-2012)
    dataTrain['month'] = dataTrain.sale_date.apply(lambda x:x.month)

    dataTrain.sort_values(by='sale_date', ascending=True, inplace=True)
    return dataTrain

def getModel(x_train,y_train,x_test=None,y_test=None):
    X_train=xgb.DMatrix(x_train,label=y_train)
    X_test=xgb.DMatrix(x_test,label=y_test)
    params = {'booster': 'gbtree',
              'objective': 'reg:logistic',
              'eval_metric': 'rmse',
              'gamma': 0.1,
              'min_child_weight': 2,
              'max_depth': 5,
              'lambda': 10,
              'alpha': 5,
              'colsample_bytree': 0.1,
              'colsample_bylevel': 0.1,
              'eta': 0.01,
              'tree_method': 'gpu_exact',
              'seed': 0,
              'gpu_id': 0,
              # 'scale_pos_weight':10,
              'nthread': -1
              }
    watchlist = [(X_train, 'train'), (X_test, 'eval')]
    model = xgb.train(params, X_train, num_boost_round=3000, evals=watchlist, early_stopping_rounds=10)
    return model


if __name__=="__main__":
    dataTrain = data_turn(dataTrain)
    # starttime=datetime.date(2012,1,1)
    # endtime=starttime.replace(year=starttime.year+2)
    # y=[]
    # train=pd.DataFrame()
    # while endtime < datetime.date(2017,10,1):
    #     data=dataTrain[(dataTrain['sale_date']>=starttime) == (dataTrain['sale_date']<=endtime)].copy()
    #     data=data.groupby(['sale_date','class_id']).agg('mean').reset_index()
    #     data.drop(['sale_date','class_id'],axis=1,inplace=True)
    #     if endtime.month==12:
    #         labeltime = endtime.replace(year=endtime.year +1)
    #         labeltime = labeltime.replace(month=1)
    #     else:
    #         labeltime = endtime.replace(month=endtime.month+1)
    #     label=dataTrain[dataTrain['sale_date']==labeltime].copy()
    #     label=label.groupby(['class_id']).agg('sum').reset_index()
    #     print(len(label))
    #     # label=label[['level_id','TR','gearbox_type','power','car_length','emission_standards_id','car_height','total_quality','equipment_quality','rated_passenger','front_track','sale_quantity']]
    #     # label.columns=
    #     train=pd.concat([train,data],axis=0)
    #     y.extend(label)
    #     print(len(label))
    #     print(len(train))
    #     break
    dataTrain.to_csv('data/train.csv', index=None)


