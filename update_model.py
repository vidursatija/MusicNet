from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import sys

class UpdateModel():

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
		self.epoch = 5
		self.avg_err = 0.0
		self.num_steps = 10

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
			input_x = self.inputXY[:, :-1]
			input_y = tf.reshape(self.inputXY[:, 1:], [-1])
			x = tf.nn.embedding_lookup(self.embeddings, input_x)
			#self.input_len = tf.placeholder(tf.int32, shape=())

		"""state0 = tf.zeros(shape=[1, 240], dtype=tf.float32)
		i0 = tf.constant(0)
		condition = lambda state, i: tf.less(i, self.input_len)
		body = lambda state[i+1], i: [tf.tanh(tf.matmul(tf.concat([x[:, i, :], state], 1), self.rnn_w) + self.rnn_b), tf.add(i, 1)]
		outputs, final_i = tf.while_loop(condition, body, loop_vars=[state0, i0], shape_invariants=[tf.TensorShape([1, 240]), tf.TensorShape([])])
		"""
		outputs = []
		state = tf.zeros(shape=[1, 240], dtype=tf.float32)
		for i in range(self.num_steps):
			state = tf.tanh(tf.matmul(tf.concat([x[:, i, :], state], 1), self.rnn_w) + self.rnn_b)
			outputs.append(state)

		output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_size])

		self.logits =  tf.matmul(output, self.rev_embed) + self.rev_bias
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], [input_y], [tf.ones_like(input_y, dtype=tf.float32)])
		self.cost = tf.reduce_sum(loss)
		learn_r = 1.0

		tvars = tf.trainable_variables()
		optimizer = tf.train.GradientDescentOptimizer(learn_r)
		gradsvars = optimizer.compute_gradients(self.cost)
		#print(gradsvars)
		grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 1)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))

		#self.sess.run(tf.global_variables_initializer())

	def train_model(self, key_queue):
		x_i = [self.key_to_int[key] for key in key_queue]
		t_cost = 100
		str_len = len(x_i) - 1
		print(str_len)
		temp_f = open("temp.train", "w")
		temp_f.close()
		for _ in range(self.epoch):
			t_cost, _ = self.sess.run([self.cost, self.train_op], feed_dict={self.inputXY: [x_i]})

		saver = tf.train.Saver()
		saver.save(self.sess, 'music_model_final')
		os.remove("temp.train")
		return t_cost

if __name__ == '__main__':
	m = UpdateModel()
	run = True
	f = dict()
	try:
		f = pickle.load(open('update_queue.p', 'rb'))
	except:
		run = False
	if run:
		q = f['q']
		print(m.train_model(q))

