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
		self.rating_items = rating_items.movie.unique()

		self.uhist = []
		for u in range(self.num_users):
			udata = rating_data[rating_data.user==u].sort_values(by=['timestamp']).values
			self.uhist.append(list(udata))

		self.test_neg= []

		with open('../data/ml-1m.test.negative','r') as f:
			for line in f.readlines():
				values = [int(i) for i in line.strip().split('\t')[1:]]
				self.test_neg.append(values)


	def _generate_neg_items(self):
		negative_items = []
		for tmp in range(self.config.neg_count):
			j = np.random.choice(self.rating_items)
			while j in self.rating_items:
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

		train_inputs = []
		train_is = []
		for index, i in enumerate(udata[window:-1]):
			train_is.append(i)
			train_inputs.append(udata[index:index+window])
		return train_inputs, train_is

	def _get_test_data(self,udata):
		len_udata = len(udata)
		window = int(len_udata*0.1)

		if window < self.config.min_window:
			window =self.config.min_window
		elif window > self.config.max_window:
			window = self.config.max_window

		test_i = udata[-1]
		test_input = udata[-1-window:-1]

		return test_input, test_i

	def getTrainBatches(self):
		u_index = list(range(self.num_users))
		np.random.shuffle(u_index)

		for u in u_index:
			input_batches = []
			item_batches = []
			len_batches = []
			label_batches = []
			udata = self.uhist[u]
			train_inputs, train_is = self._get_train_data(udata)
			for train_input, train_i in  zip(train_inputs, train_is):
				input_batches.append(train_input)
				item_batches.append(train_i)
				len_batches.append(len(train_input))
				label_batches.append(1)
				for neg_i in self._generate_neg_items(udata):
					input_batches.append(train_input)
					item_batches.append(neg_i)
					len_batches.append(len(train_input))
					label_batches.append(0)

			yield input_batches, item_batches, len_batches, label_batches



	def getTestBatches(self):
		u_index = list(range(self.num_users))
		np.random.shuffle(u_index)

		for u in u_index:
			input_batches = []
			item_batches = []
			len_batches = []
			label_batches = []
			udata = self.uhist[u]
			neg_items = self.neg_items[u]

			test_input, test_i =  self._get_test_data(udata)
			for neg_i in neg_items:
				input_batches.append(test_input)
				item_batches.append(neg_i)
				len_batches.append(len(test_input))
				len_batches.append(0)
			input_batches.append(test_input)
			item_batches.append(test_i)
			len_batches.append(len(test_input))
			len_batches.append(1)

			yield input_batches, item_batches, len_batches, label_batches





