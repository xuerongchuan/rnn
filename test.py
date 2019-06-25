from config import Config
from rnnRS import RnnRs
from reader import Dataloader

test= 0
config = Config()
if test:
	config.data_path = 'data/ml-1m/sample.csv'
	config.num_users = 50

dl = Dataloader(config)
model = RnnRs(config, dl)
model.train_and_evaluate()