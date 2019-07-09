import pandas as pd
import numpy as np
import json


class Dataloader(object):

	def __init__(self, config):
		self.config = config
		self.num_items = 3883
		self.num_users = self.config.num_users
		#self._init_data()

		

	

	def getTrainShuffleBatches(self):
		with open(self.config.train_path) as f:
			data = f.readline()
			batch_i = 0
			user_data = []
			input_data = []
			item_data = []
			label_data = []
			len_data = []
			while data:
				data = json.loads(data)
				user_data.append(data[0])
				input_data.append(np.array(data[1]))
				item_data.append(data[2])
				label_data.append(data[4])
				len_data.append(data[3])
				batch_i +=1
				data = f.readline()
				if batch_i > self.config.batch_size:
					indexes = list(range(len(user_data)))
					np.random.shuffle(indexes)
					user_data = [user_data[i] for i in indexes]
					input_data = [input_data[i] for i in indexes]
					item_data = [item_data[i] for i in indexes]
					label_data = [label_data[i] for i in indexes]
					len_data = [len_data[i] for i in indexes]
					yield  input_data, item_data, len_data,label_data, user_data
					user_data = []
					input_data = []
					item_data = []
					label_data = []
					len_data = []
					batch_i = 0
				
	def getTestBatches(self):
		with open(self.config.test_path) as f:
			data = f.readline()
			batch_i = 0
			user_data = []
			input_data = []
			item_data = []
			label_data = []
			len_data = []
			while data:
				data = json.loads(data)
				user_data.append(data[0])
				input_data.append(np.array(data[1]))
				item_data.append(data[2])
				label_data.append(data[4])
				len_data.append(data[3])
				batch_i +=1
				data = f.readline()
				if batch_i > self.config.batch_size:
					indexes = list(range(len(user_data)))
					np.random.shuffle(indexes)
					user_data = [user_data[i] for i in indexes]
					input_data = [input_data[i] for i in indexes]
					item_data = [item_data[i] for i in indexes]
					label_data = [label_data[i] for i in indexes]
					len_data = [len_data[i] for i in indexes]
					yield input_data, item_data, len_data,label_data, user_data
					user_data = []
					input_data = []
					item_data = []
					label_data = []
					len_data = []
					batch_i = 0
				
					
				

