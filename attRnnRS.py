from rnnRS import RnnRs

class AttRnnRs(RnnRS):
	def __init__(self, config, dl):
		super(AttRnnRs, self).__init__(config, dl)

	def _attention_model(self, p_, q)

