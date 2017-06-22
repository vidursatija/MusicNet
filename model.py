from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os

tf.logging.set_verbosity(tf.logging.INFO)

#BG = BatchGenerator(batch_size, num_steps, windows_size, vocab_size)
sess = tf.Session()

input_file = tf.read_file(os.path.join("", 'input.p'))
input_file_data = pickle.loads(sess.run(input_file))
int_to_key = input_file_data['itk']
key_to_int = input_file_data['kti']
all_songs = input_file_data['ixx']

feature_file = tf.read_file(os.path.join("", 'songsf.p'))
all_features = pickle.loads(sess.run(feature_file))
int_to_features = np.empty((len(all_features)+1, 240))
int_to_features[0] = np.zeros((240))

for key, feature in all_features.items():
	int_to_features[key_to_int[int(key)]] = feature

hidden_size = 240
vocab_size = len(all_features) + 1
batch_size = 1
#num_steps = 31
#windows_size = 15
n_files = len(all_songs)
epoch = 1600*n_files
learn_rate = 3.0
avg_err = 0.0

embeddings = tf.Variable(int_to_features, dtype=tf.float32, trainable=False)

rnn_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicRNNCell(hidden_size), output_keep_prob=0.90)#tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85), tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=0.85)])#tf.contrib.rnn.GRUCell(hidden_size)
rev_embed = tf.Variable(tf.truncated_normal([hidden_size, vocab_size], stddev=0.01, dtype=tf.float32), name='rev_w')
rev_bias = tf.Variable(tf.constant(0.1, shape=[vocab_size], dtype=tf.float32), name='rev_b')

with tf.device("/cpu:0"):
	inputXY = tf.placeholder(tf.int32, shape=[1, None])
	input_y = tf.reshape(inputXY[:, 1:], [-1])
	input_x = inputXY[:, :-1]

	x = tf.nn.embedding_lookup(embeddings, input_x)
	x = tf.nn.dropout(x, 0.90)


with tf.variable_scope("RNN"):
	outputs, f_state = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)


output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

logits =  tf.matmul(output, rev_embed) + rev_bias
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [input_y], [tf.ones_like(input_y, dtype=tf.float32)])
cost = tf.reduce_sum(loss) / batch_size
learn_r = tf.Variable(learn_rate, trainable=False)

tvars = tf.trainable_variables()
optimizer = tf.train.GradientDescentOptimizer(learn_r)
gradsvars = optimizer.compute_gradients(cost)
#print(gradsvars)
grads, _ = tf.clip_by_global_norm([g for g, v in gradsvars], 1)#tf.clip_by_global_norm(tf.gradients(cost, tvars), 10)
train_op = optimizer.apply_gradients(zip(grads, tvars))
new_lr = tf.placeholder(tf.float32, shape=[])
lr_update = tf.assign(learn_r, new_lr)
saver = tf.train.Saver()


sess.run(tf.global_variables_initializer())
"""variables_names = [v.name for v in tvars]
sess.run(variables_names)
for k in variables_names:
	print(k)"""

for f in range(epoch):
	#sess.run(state_assign)
	cost_eval, _ = sess.run([cost, train_op], feed_dict={inputXY: [all_songs[f%n_files]]})
	avg_err = (avg_err*(f) + cost_eval)/(f+1)
	if f%800 == 799:
		sess.run(lr_update, feed_dict={new_lr: learn_rate/(1+(0.000025*f))})
	if f%500 == 1:
		print ("Ep: "+str(f)+"Avg:"+str(avg_err))

	if f%20000 == 19999:
		saver.save(sess, 'rnn_rel', global_step=f)
print(avg_err)
saver.save(sess, 'music_model_final')

#print(sess.run(logits, feed_dict))
#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
