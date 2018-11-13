import pandas as pd, collections



class Load_data:
	def __init__(self, batch_size, sequence_length):
		self.batch_size = batch_size
		self.sequence_length = sequence_length


	def prepare_data(self):
		self.x_train = []; self.y_train = []
		with open('train_corpus.tsv') as f:
			for line in f.read().strip().split('\n'):
				if len(line.split('\t')) == 2:
					[x,y] = line.split('\t')
					self.x_train.append(x)
					self.y_train.append(self.fix_class(y))

		self.x_val = []; self.y_val = []
		with open('test_corpus.tsv') as f:
			for line in f.read().strip().split('\n'):
				if len(line.split('\t')) == 2:
					[x,y] = line.split('\t')
					self.x_val.append(x)
					self.y_val.append(self.fix_class(y))
		self.x_train = self.convert_word_num(self.x_train)
		self.x_val = self.convert_word_num(self.x_val)		



	def convert_word_num(self, x):
		dictionary = list(set(self.x_train))
		self.word_to_int = {item: index+1 for index, item in enumerate(dictionary)}
		for index, item in enumerate(x):
			x[index] = self.word_to_int.get(item, 0)
		return x


	def fix_class(self, y):
		class_map = {
			'0': 0, 
			'PER': 1,
			'ORG': 2,
			'LOC': 3,
			'MISC': 4,
			'PRG': 5}
		return class_map[y]


	def get_train_batch(self, i):
		if (i+1)*self.batch_size < len(self.x_train):
			return self.x_train[i*self.batch_size : (i+1)*self.batch_size], self.y_train[i*self.batch_size : (i+1)*self.batch_size] 
		else:
			return self.x_train[i*self.batch_size :], self.y_train[i*self.batch_size :] 


	def get_val_batch(self, i):
		if (i+1)*self.batch_size < len(self.x_val):
			return self.x_val[i*self.batch_size : (i+1)*self.batch_size], self.y_val[i*self.batch_size : (i+1)*self.batch_size] 
		else:
			return self.x_val[i*self.batch_size :], self.y_val[i*self.batch_size :]




if __name__ == '__main__':
	l = Load_data(128, 100)
	l.prepare_data()
	print(l.get_train_batch(10))










