from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os

class PredictModel():

	def __init__(self, key_file_name='input.p', feature_file_name='songsf.p'):
		self.sess = tf.Session()

		input_file = tf.read_file(os.path.join("", key_file_name))
		input_file_data = pickle.loads(self.sess.run(input_file))
		self.int_to_key = input_file_data['itk']
		self.key_to_int = input_file_data['kti']

		feature_file = tf.read_file(os.path.join("", feature_file_name))
		all_features = pickle.loads(self.sess.run(feature_file))

		self.hidden_size = 240
		self.vocab_size = len(all_features) + 1
		self.epoch = 10
		self.learn_rate = 3.0
		self.avg_err = 0.0

		saver = tf.train.import_meta_graph('music_model_final.meta')
		saver.restore(self.sess,tf.train.latest_checkpoint('./'))
		all_vars = tf.global_variables()

		self.embeddings = all_vars[0]#tf.get_variable('Variable', shape=[self.vocab_size, 240], dtype=tf.float32, trainable=False)

		self.rnn_w =  all_vars[3]#tf.get_variable('RNN/rnn/basic_rnn_cell/weights:0', shape=[480, 240], dtype=tf.float32)#, [240, 240], 0)
		self.rnn_b =  all_vars[4]#tf.get_variable('RNN/rnn/basic_rnn_cell/biases:0', shape=[240], dtype=tf.float32)
		self.rev_embed = all_vars[1]#tf.get_variable('rev_w:0', shape=[240, self.vocab_size], dtype=tf.float32)
		self.rev_bias = all_vars[2]#tf.get_variable('rev_b:0', shape=[self.vocab_size], dtype=tf.float32)

		with tf.device("/cpu:0"):
			self.inputXY = tf.placeholder(tf.int32, shape=[1, None])
			self.inputXY = self.inputXY[:, -10:]
			x = tf.nn.embedding_lookup(self.embeddings, self.inputXY)
			self.input_len = tf.placeholder(tf.int32, shape=())

		state0 = tf.zeros(shape=[1, 240], dtype=tf.float32)
		i0 = tf.constant(0)
		condition = lambda state, i: tf.less(i, self.input_len)
		body = lambda state, i: [tf.tanh(tf.matmul(tf.concat([x[:, i, :], state], 1), self.rnn_w) + self.rnn_b), tf.add(i, 1)]
		outputs, final_i = tf.while_loop(condition, body, loop_vars=[state0, i0], shape_invariants=[tf.TensorShape([1, 240]), tf.TensorShape([])])

		output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_size])

		self.logits =  tf.matmul(output, self.rev_embed) + self.rev_bias


		#self.sess.run(tf.global_variables_initializer())

	def predictNext(self, key_queue=['-1']):
		x_i = [self.key_to_int[key] for key in key_queue]
		probs = self.sess.run(self.logits, feed_dict={self.inputXY: [x_i], self.input_len:len(x_i)})
		top3 = probs[0].argsort()[-3:][::-1]
		return [self.int_to_key[int(top)] for top in top3]


