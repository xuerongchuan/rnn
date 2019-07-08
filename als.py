import pandas as pd
import numpy as np
import time
import scipy.sparse as sp
#import pandas as pd
from implicit.als import AlternatingLeastSquares
import math
from  sklearn.metrics import roc_auc_score 

train = pd.read_csv('../data/ml-1m/train_month.csv')
test = pd.read_csv('../data/ml-1m/test_month.csv')


train_sp_matrix = sp.coo_matrix(([1]*len(train), [train.user.values, \
			train.movie.values])).tolil()
item_list = train.movie.unique()
for u in train.user:
	user_hist = train[train.user==u].movie.values
	neg_i = np.random.choice(item_list)
	while neg_i in user_hist:
		neg_i = np.random.choice(item_list)
	train_sp_matrix[u, neg_i] = 0
train_sp_matrix =train_sp_matrix.tocsr()
print('开始构建模型')
model = AlternatingLeastSquares(factors= 16, iterations = 100)
model.fit(train_sp_matrix.transpose()) 
print('模型构建完毕')
aucs = []
for u in test.user.unique():
	user_hist = train[train.user==u].movie.values
	labels = []
	predictions = []
	items = list(test[test.user==u].movie.values)
	neg_items = []
	labels = [1]*len(items)
	for i in items:
		neg_i = np.random.choice(item_list)
		while neg_i in user_hist:
			neg_i = np.random.choice(item_list)
		neg_items.append(neg_i)
		labels.append(0)
	items = items + neg_items


	user_factor = model.user_factors[u].reshape((1,16))
	item_factors = model.item_factors[items]
	predictions = np.dot(user_factor, item_factors.transpose()).reshape(-1)
	auc = roc_auc_score(predictions, labels)
	aucs.append(auc)
auc = np.mean(aucs)
print('aucs: %.4f'%auc)
	


