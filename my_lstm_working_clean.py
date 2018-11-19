from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print("\n\n")
print("\n\n")
print("\n\n")
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input], name="X_plc")
Y = tf.placeholder("float", [None, num_classes], name="Y_plc")

n_h1 = 20
n_h2 = 15

def RNN(x):

    n_cells_layers = [n_h1, n_h2]
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_) for n_ in n_cells_layers]
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=x, dtype=tf.float32, scope = "dynamic_rnn")

    fc = tf.contrib.layers.fully_connected(outputs[:,-1], num_classes, activation_fn = None, scope="my_fully_connected")
    return fc

logits = RNN(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name = "Adam")
loss_op = tf.reduce_mean(xentropy)
train_op = optimizer.minimize(loss_op, name = "train_op")
tf.summary.scalar("loss", loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('tensorboard/2')
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    writer.add_graph(sess.graph)
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        if step % 10 == 0:
            summary = sess.run(merged, feed_dict = {X: batch_x, Y: batch_y})
            writer.add_summary(summary, step)
            # writer.flush()

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128*2
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
