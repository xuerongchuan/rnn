from config import Config
from rnnRS import RnnRs
from userRnn import UserRNN
from reader import Dataloader

test= 0
user = 0
config = Config()
if test:
	config.data_path = '../data/ml-1m/sample.csv'
	config.num_users = 50



dl = Dataloader(config)
if user:
	model = UserRNN(config, dl)
else:
	model = RnnRs(config, dl)
model.train_and_evaluate()