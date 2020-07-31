import tensorflow as tf
import numpy as np
from tqdm import tqdm
mnist=tf.keras.datasets.mnist

def to_one_hot(dimension,tensor):
	rtn=[]
	for i in tensor:
		vec=np.zeros(dimension)
		vec[i]=1
		rtn.append(vec)
	return np.array(rtn)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
plt.figure("fig")
for i in range(10):
	plt.subplot(2,5,i+1)
	plt.imshow(x_train[i],cmap='gray')
plt.show()

x_test, x_train = x_test / 255., x_train / 255.
x_test=np.reshape(x_test,newshape=[-1,28,28,1])
x_train=np.reshape(x_train,newshape=[-1,28,28,1])
y_test, y_train = to_one_hot(10, y_test), to_one_hot(10, y_train)



train_size=x_train.shape[0]
test_size=x_test.shape[0]

BATCH_SIZE = 100
LEARNING_RATE = 0.0001
EPOCHS = 20

x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)

flatten=tf.layers.flatten(x)
hidden=tf.layers.dense(flatten,units=256,activation=tf.nn.relu)
logits=tf.layers.dense(hidden,units=10)

loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(y,logits))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correctPredictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, "float"))

init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epochs in range(EPOCHS):
		num_iters = (train_size + BATCH_SIZE - 1) // BATCH_SIZE
		with tqdm(total=num_iters, desc="epoch %3d" % (epochs + 1), ascii=True) as pbar:
			for iter in range(num_iters):
				batch_x = x_train[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE, :, :]
				batch_y = y_train[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE, :]
				_, ls, acc = sess.run([optimizer, loss, accuracy],
									  feed_dict={x: batch_x, y: batch_y})
				pbar.set_postfix_str(
					"loss: %.6f, accuracy:%.2f" % (ls, acc))
				pbar.update(1)
	num_iters = (test_size + BATCH_SIZE - 1) // BATCH_SIZE
	with tqdm(total=num_iters, desc="testing", ascii=True) as pbar:
		ls_total = 0.
		acc_total = 0.
		for iter in range(num_iters):
			batch_x = x_test[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE, :, :]
			batch_y = y_test[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE, :]
			ls, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
			ls_total += ls
			acc_total += acc
			pbar.update(1)
		acc_total /= num_iters
		ls_total /= num_iters
		pbar.set_postfix_str("loss: %.6f, accuracy:%.2f" % (ls_total, acc_total))