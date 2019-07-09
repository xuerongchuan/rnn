from config import Config


from aucReader import Dataloader


test= 0
user = 0
als = 0
attention = 1
config = Config()
if test:
	config.data_path = '../data/ml-1m/sample.csv'
	config.num_users = 50
	config.train_path = '../data/ml-1m/train2'



dl = Dataloader(config)
if user:
	from userRnn import UserRNN
	print('user model test:%d'%test)
	model = UserRNN(config, dl)
elif attention:

	from attURnn import AttUserRNN
	print('attention model test:%d'%test)
	model = AttUserRNN(config, dl)
elif als:
	from als import ALS
	print('als model test:%d'%test)
	model = ALS(config, dl)
else:
	from rnnRS import RnnRs
	print('rnn model test:%d'%test)
	model = RnnRs(config, dl)
model.train_and_evaluate()