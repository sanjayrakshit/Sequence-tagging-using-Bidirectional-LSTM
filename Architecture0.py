import tensorflow as tf 
from loaddata import *
from config import *


l = Load_data(batch_size, sequence_length)
l.prepare_data()

vocabulary_size = len(set(l.x_train)) + 1

def one_hot(y):
	return tf.one_hot(y, num_class)


def make_lstm_cell(rnn_cell_size, keep_prob):
    lstm = tf.nn.rnn_cell.LSTMCell(rnn_cell_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop


def build_rnn():
	tf.reset_default_graph()

	with tf.name_scope('inputs'):
		inputs = tf.placeholder(tf.int32, [None, sequence_length], 'inputs')

	with tf.name_scope('labels'):
		labels = tf.placeholder(tf.float32, [None, sequence_length, num_class], 'labels')

	keep_prob_rnn = tf.placeholder(tf.float32, name='keep_prob_rnn')
	keep_prob_dense = tf.placeholder(tf.float32, name='keep_prob_dense')

	with tf.name_scope("embeddings"):
		embedding = tf.Variable(tf.truncated_normal((vocabulary_size, word_embedding_size), -0.1, 0.1))
		embed = tf.nn.embedding_lookup(embedding, inputs)

    # Build the RNN layers
	with tf.name_scope("RNN_fw_layers"):
		cell_fw = tf.contrib.rnn.MultiRNNCell([make_lstm_cell(rnn_cell_size, keep_prob_rnn) for _ in range(rnn_layer_size)])

	with tf.name_scope("RNN_bw_layers"):
		cell_bw = tf.contrib.rnn.MultiRNNCell([make_lstm_cell(rnn_cell_size, keep_prob_rnn) for _ in range(rnn_layer_size)])
    
    # Set the initial state
	with tf.name_scope("RNN_fw_init_state"):
		initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)

	with tf.name_scope("RNN_bw_init_state"):
		initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

	# Run the data through the RNN layers
	with tf.name_scope("Bi_RNN"):
		outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embed, 
			initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)
		(output_fw, output_bw) = outputs

	print(type(output_fw), type(output_bw))



build_rnn()
