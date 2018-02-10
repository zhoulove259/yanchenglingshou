import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# def getdata(data):
#     for id in data.class_id.unique():
#         data.loc[data['class_id'] == id, 'Y']=data.loc[data['class_id']==id,'sale_quantity'].shift(-1)
#     return data.dropna()
#
# data=pd.read_csv('data/train.csv')
#
# datainfo=data[data.columns.difference(['sale_date','sale_quantity'])]
# datainfo=datainfo.groupby(['class_id']).agg('mean').reset_index()
# print(len(datainfo))
#
# data=data[['sale_date','class_id','sale_quantity']]
# data=data.groupby(['sale_date','class_id']).agg('sum').reset_index()
# data['Y']=0
# data =getdata(data)
# data=pd.merge(data,datainfo,on=['sale_date','class_id'],how='left')
# y=data['Y']
# X=data.drop(['Y','class_id','sale_date'],axis=1)


data=pd.read_csv('data/train.csv')
tmp=data[data['sale_date']=='2017-10-01'].copy()
print(tmp)
y=data['sale_quantity']
X=data.drop(['sale_quantity','sale_date','class_id'],axis=1)



scaler=MinMaxScaler()
scaler.fit(y)
y=scaler.transform(y)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3, random_state=0)


X=xgb.DMatrix(X,label=y)
X_train = xgb.DMatrix(X_train,label=Y_train)
X_test = xgb.DMatrix(X_test,label=Y_test)

params = {'booster': 'gbtree',
          'objective': 'reg:logistic',
          'eval_metric': 'rmse',
          'gamma': 0,
          'min_child_weight': 2,
          'max_depth': 5,
          'lambda': 4,
          'alpha': 0.3,
          'silent':True,
          'subsample ':0.8,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.8,
          'eta': 0.03,
          'tree_method': 'gpu_exact',
          'seed': 50,
          'gpu_id': 0,
          # 'scale_pos_weight':10,
          'nthread': -1
          }
watchlist = [(X_train, 'train'),(X_test,'eval')]
model = xgb.train(params, X_train, num_boost_round=3000, evals=watchlist,early_stopping_rounds=10)
model.save_model('0001.model')

preds=model.predict(X_train)
preds=scaler.inverse_transform(preds)
Y_train=scaler.inverse_transform(Y_train)
print('train error:',np.sqrt(mean_squared_error(preds,Y_train)))

preds=model.predict(X_test)
preds=scaler.inverse_transform(preds)
Y_test=scaler.inverse_transform(Y_test)
print('test error:',np.sqrt(mean_squared_error(preds,Y_test)))

feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []
for (key, value) in feature_score:
    fs.append("{0},{1}\n".format(key, value))

with open('xgb_feature_score.csv', 'w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


model = xgb.train(params, X, num_boost_round=2600,early_stopping_rounds=10,evals=watchlist)

test=tmp.drop(['sale_date','class_id','sale_quantity'],axis=1)
test['month']=11
preds=model.predict(xgb.DMatrix(test))
preds=scaler.inverse_transform(preds)
tmp['predict_quantity']=preds
tmp['predict_date']='201711'
tmp=tmp[['predict_date','class_id','predict_quantity']]
tmp=tmp.groupby(['predict_date','class_id']).agg('sum').reset_index()
tmp.to_csv('result.csv',index=False,header=True)
print(tmp)
print(tmp.describe())
print(tmp.info())
