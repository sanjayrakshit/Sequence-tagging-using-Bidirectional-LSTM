import pandas as pd 



class Load_data:
	def __init__(self, batch_size, sequence_length):
		self.batch_size = batch_size
		self.sequence_length = sequence_length
		self.train = pd.read_csv("train_corpus.tsv", delimiter="\t")
		self.test = pd.read_csv("test_corpus.tsv", delimiter="\t")


	def prepare_data(self):
		pass