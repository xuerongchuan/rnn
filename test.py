from config import Config
from rnnRS import RnnRs
from reader import Dataloader

config = Config()
dl = Dataloader(config)
model = RnnRs(config, dl)
model.train_and_evaluate()