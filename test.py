from config import Config
from rnnRS import RnnRs
from userRnn import UserRNN
from attURnn import AttUserRNN
from reader import Dataloader

test= 0
user = 0
attention = 1
config = Config()
if test:
	config.data_path = '../data/ml-1m/sample.csv'
	config.num_users = 50



dl = Dataloader(config)
if user:
	print('user model test:%d'%test)
	model = UserRNN(config, dl)
elif attention:
	print('attention model test:%d'%test)
	model = AttUserRNN(config, dl)
else:
	print('rnn model test:%d'%test)
	model = RnnRs(config, dl)
model.train_and_evaluate()