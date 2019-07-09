# -*- coding: utf-8 -*-

import numpy as np 
import scipy.sparse as sp
#import pandas as pd
from implicit.als import AlternatingLeastSquares
import math
from  sklearn.metrics import roc_auc_score 

class ALS(object):
	def __init__(self,config, dl):
		self.config = config
		self.dl = dl
		self.train = self.get_trainmat()

#    def build_model(self):
#        print('creating data matrix...')
#        
#        test = sp.lil_matrix(train_data.shape)
#        for u, uData in enumerate(self.dl.testset()):
#            items, _  = zip(*uData)
#            for i in items:
	def get_trainmat(self):

		train = sp.lil_matrix((self.dl.num_users, self.dl.num_items))
		for u,uData in enumerate(self.dl.trainset()):
			items, _  = zip(*uData)
			for i in items:
				train[u,i] = 1
				for tmp in range(self.config.neg_count):
					j = np.random.choice(self.dl.num_items)
					while j in items:
						j = np.random.choice(self.dl.num_items)
					train[u,j] = 0
		return train.tocsr()
	def evaluate(self):
		hits, ndcgs,precisions, recalls,aucs = [],[],[],[],[]


		for batches in self.dl.getTestBatches():

			u = batches[4][0]
			items = batches[1]
			labels = np.array(batches[3])

			items = np.array(items, dtype = np.int64)
			user_factor = self.model.user_factors[u]
			item_factors = self.model.item_factors[items]
			predictions = np.dot(user_factor, item_factors.transpose()).reshape(-1)
			auc = roc_auc_score( labels,predictions)
			predictions = predictions > 0.5
	        precision = sum(predictions*labels)/sum(predictions)
	        recall = sum(predictions*labels)/sum(labels)
	        predictions.append(precision)
	        recalls.append(recall)
	       	aucs.append(auc)
		# 	neg_predict, pos_predict = np.array(predictions[:-1]), np.array(predictions[-1])
		# 	position = (neg_predict >= pos_predict).sum()
		# 	#print(position)
		# 	hr = position < 10
		# 	ndcg = math.log(2) / math.log(position+2) if hr else 0
		# 	hits.append(hr)
		# 	ndcgs.append(ndcg)  
		# hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
		precision = np.mean(predictions)
		recall = np.mean(recalls)
		auc = np.mean(aucs)
		print('precisione:{0}, recall:{1}, auc:{2}, '.format(precision, recall, auc))
            
	def train_and_evaluate(self):
		self.model = AlternatingLeastSquares(factors= 16, \
		                                iterations = 100)
		print('model is going to fit!')
		self.model.fit(self.train.transpose())
		print('model is already!')
		self.evaluate()

                    
        


# import pandas as pd
# import numpy as np
# import time
# import scipy.sparse as sp
# #import pandas as pd
# from implicit.als import AlternatingLeastSquares
# import math
# from  sklearn.metrics import roc_auc_score 

# train = pd.read_csv('../data/ml-1m/train_month.csv')
# test = pd.read_csv('../data/ml-1m/test_month.csv')


# train_sp_matrix = sp.coo_matrix(([1]*len(train), [train.user.values, \
# 			train.movie.values])).tolil()
# item_list = train.movie.unique()
# for u in train.user:
# 	user_hist = train[train.user==u].movie.values
# 	neg_i = np.random.choice(item_list)
# 	while neg_i in user_hist:
# 		neg_i = np.random.choice(item_list)
# 	train_sp_matrix[u, neg_i] = 0
# train_sp_matrix =train_sp_matrix.tocsr()
# print('开始构建模型')
# model = AlternatingLeastSquares(factors= 16, iterations = 100)
# model.fit(train_sp_matrix.transpose()) 
# print('模型构建完毕')
# aucs = []
# for u in test.user.unique():
# 	user_hist = train[train.user==u].movie.values
# 	labels = []
# 	predictions = []
# 	items = list(test[test.user==u].movie.values)
# 	neg_items = []
# 	labels = [1]*len(items)
# 	for i in items:
# 		neg_i = np.random.choice(item_list)
# 		while neg_i in user_hist:
# 			neg_i = np.random.choice(item_list)
# 		neg_items.append(neg_i)
# 		labels.append(0)
# 	items = items + neg_items


# 	user_factor = model.user_factors[u].reshape((1,16))
# 	item_factors = model.item_factors[items]
# 	predictions = np.dot(user_factor, item_factors.transpose()).reshape(-1)
# 	auc = roc_auc_score(predictions, labels)
# 	aucs.append(auc)
# auc = np.mean(aucs)
# print('aucs: %.4f'%auc)
	


