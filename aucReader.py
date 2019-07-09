import pandas as pd
import numpy as np


class Dataloader(object):

	def __init__(self, config):
		self.config = config
		self.num_items = 3883
		self.num_users = self.config.num_users
		self._init_data()

	def _init_data(self):
		print('loading data ...')
		rating_data = pd.read_csv(self.config.data_path)
		self.rating_items = list(rating_data.movie.unique())

		self.uhist = []
		for u in range(self.num_users):
			udata = rating_data[rating_data.user==u].sort_values(by=['timestamp']).movie.values
			self.uhist.append(list(udata))

		self.test_neg= []
		

		with open('../data/ml-1m.test.negative','r') as f:
			for line in f.readlines():
				values = [int(i) for i in line.strip().split('\t')[1:]]
				self.test_neg.append(values)

		print('data has been already!!!!!')
	def trainset(self):
		for udata in self.uhist:
			yield zip(udata[:-10], [0]*len(udata[:-10]))

	def _generate_neg_items(self, udata):
		negative_items = []
		for tmp in range(self.config.neg_count):
			j = np.random.choice(self.rating_items)
			while j in udata:
				j = np.random.choice(self.rating_items)
			#jt = self.dl.ITmap[str(j)] if str(j) in self.dl.ITmap.keys() else self.dl.numT
			negative_items.append(j)
		return negative_items

	def _get_train_data(self, udata):
		len_udata = len(udata)
		window = int(len_udata*0.1)

		if window < self.config.min_window:
			window = self.config.min_window
		elif window > self.config.max_window:
			window = self.config.max_window
		if self.config.window:
			window = self.config.window

		train_inputs = []
		train_is = []
		for index, i in enumerate(udata[window:]):
			train_is.append(i)
			train_inputs.append(udata[index:index+window])
		return train_inputs, train_is
	def _get_train_shuffle_data(self, u, udata):

		user_inputs, user_is = self._get_train_data(udata)
		return list(zip([u]*len(user_inputs), user_inputs, user_is))
	def _getTrainShuffle(self):
		all_inputs = []
		for u in range(self.num_users):
			udata = self.uhist[u]
			udata = udata[-self.config.batch_len-1:-10]
			all_inputs.append(self._get_train_shuffle_data(u,udata))
		all_inputs = sum(all_inputs,[])
		return all_inputs
	def getTrainShuffleBatches(self):
		all_inputs = self._getTrainShuffle()
		lenth = len(all_inputs)
		batch_len = lenth // self.config.batch_size
		for batch_i in range(batch_len):
			data = all_inputs[batch_i*self.config.batch_size: batch_i*self.config.batch_size+self.config.batch_size] 
			yield self._get_shuffle_batches(data)
	def _get_shuffle_batches(self, data):
		np.random.shuffle(data)
		user_batches = []
		input_batches = []
		item_batches = []
		len_batches = []
		label_batches = []
		for u, u_input, ui in data:
			user_batches.append(u)
			input_batches.append(u_input)
			item_batches.append(ui)
			len_batches.append(len(u_input))
			label_batches.append(1)
			for neg_i in self._generate_neg_items(self.uhist[u]):
					input_batches.append(u_input)
					item_batches.append(neg_i)
					len_batches.append(len(u_input))
					label_batches.append(0)
					user_batches.append(u)
		return input_batches, item_batches, len_batches, label_batches, user_batches


		

		

	def _get_test_data(self,udata):
		len_udata = len(udata)
		window = int(len_udata*0.1)

		if window < self.config.min_window:
			window =self.config.min_window
		elif window > self.config.max_window:
			window = self.config.max_window

		if self.config.window:
			window = self.config.window

		test_is = udata[-10:]
		test_input = udata[-10-window:-10]

		return test_input, test_is

	def getTrainBatches(self):
		u_index = list(range(self.num_users))
		np.random.shuffle(u_index)

		for u in u_index:
			user_batches = []
			input_batches = []
			item_batches = []
			len_batches = []
			label_batches = []
			udata = self.uhist[u]
			udata = udata[-self.config.batch_len-1:-1]
			train_inputs, train_is = self._get_train_data(udata)
			for train_input, train_i in  zip(train_inputs, train_is):
				user_batches.append(u)
				input_batches.append(train_input)
				item_batches.append(train_i)
				len_batches.append(len(train_input))
				label_batches.append(1)
				for neg_i in self._generate_neg_items(udata):
					input_batches.append(train_input)
					item_batches.append(neg_i)
					len_batches.append(len(train_input))
					label_batches.append(0)
					user_batches.append(u)

			yield input_batches, item_batches, len_batches, label_batches, user_batches



	def getTestBatches(self):
		u_index = list(range(self.num_users))
		np.random.shuffle(u_index)

		for u in u_index:
			user_batches = []
			input_batches = []
			item_batches = []
			len_batches = []
			label_batches = []
			udata = self.uhist[u]
			neg_items = np.random.choice(self.test_neg[u],10)

			test_input, test_is =  self._get_test_data(udata)
			for neg_i in neg_items:
				input_batches.append(test_input)
				item_batches.append(neg_i)
				len_batches.append(len(test_input))
				label_batches.append(0)
				user_batches.append(u)
			for i in test_is:
				input_batches.append(test_input)
				item_batches.append(i)
				len_batches.append(len(test_input))
				label_batches.append(1)
				user_batches.append(u)

			yield input_batches, item_batches, len_batches, label_batches, user_batches





