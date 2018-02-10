import pandas as pd
import numpy as np
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation

from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

def getmodel(shape1,shape2):
    model = Sequential()
    model.add(LSTM(32, input_shape=(shape1,shape2)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation("tanh"))
    model.compile(loss='mse', optimizer='adam')
    return model

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

dataTrain=pd.read_csv('data/train.csv')
datainfo=dataTrain[dataTrain.columns.difference(['sale_date','sale_quantity'])]
datainfo=datainfo.groupby(['class_id']).agg('mean').reset_index()
print(len(datainfo))

dataTrain=dataTrain[['sale_date','class_id','sale_quantity']]
dataTrain=dataTrain.groupby(['sale_date','class_id']).agg('sum').reset_index()
dataTrain.sort_values(by='sale_date',ascending=True,inplace=True)
dataTrain=dataTrain[['class_id','sale_quantity']]
result=0
count=1
step=12
v=[]
for id in dataTrain['class_id'].unique():
	print(id)
	tmp=dataTrain[dataTrain['class_id']==id]
	tmp=tmp.sale_quantity.values
	print(tmp)
	tmp=np.log1p(tmp)
	scaler = MinMaxScaler(feature_range=(0, 1))
	tmp=scaler.fit_transform(tmp.reshape(-1,1))
	t=tmp
	tmp = series_to_supervised(list(tmp),step, 1).values
	k = 1
	while len(tmp) == 0:
		tmp = series_to_supervised(list(t), step - k, 1).values
		k = k + 1
	tmp = series_to_supervised(list(t), step - k, 1).values
	if len(t) < 3:
		tmp = np.asarray(t).reshape(1, len(t))
	train = tmp[:-1, :]
	test = tmp[-1:, :]
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]

	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	model=getmodel(train_X.shape[1], train_X.shape[2])
	history = model.fit(train_X, train_y, epochs=50, batch_size=5, verbose=0, shuffle=False)
	true=np.expm1(scaler.inverse_transform(test_y).reshape(-1,1)[0][0])
	pred=np.expm1(scaler.inverse_transform( model.predict(test_X).reshape(-1,1))[0][0])
	v.append([id, pred])
	result=result+(pred-true)**2
	print(np.sqrt(result /count))
	print('pred:',pred)
	print('true:', true)
	print(count)
	count=count+1
print(np.sqrt(result/len(dataTrain['class_id'].unique())))
# pyplot.plot(dataTrain[dataTrain['class_id']==289403].sale_quantity.values, label='preds')
# # pyplot.plot(test_y, label='true')
# pyplot.legend()
# pyplot.show()
d=pd.DataFrame(v)
d.columns=['class_id','sale_quantity']
d.to_csv('lstmpred.csv',index=None)