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
		# Look at the explanation in the link below
		# https://stackoverflow.com/questions/49242266/difference-between-multirnncell-and-stack-bidirectional-dynamic-rnn-in-tensorflo
		o, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embed, 
			initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw)
		(output_fw, output_bw) = o

	with tf.name_scope('Stacking'):
		# We are going to stack the 
		output = tf.concat([output_fw, output_bw], 2)

	with tf.name_scope('Pre-Dense_layer'):
		output = tf.reshape(output, [-1, output.get_shape()[2]])
	
	print(output.get_shape())

	labels = tf.reshape(labels, [-1, num_class])

	with tf.name_scope('cost'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradients(labels), logits=output))

		tf.summary.scalar('cost', cost)

	with tf.name_scope('accuracy'):
		corr_pred = tf.equals(tf.argmax(output, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

		tf.summary.scalar('accuracy', accuracy)


		
build_rnn()
