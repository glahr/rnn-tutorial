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

# # Define weights
# weights = {
#     'out': tf.Variable(tf.random_normal([n_h2, num_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([num_classes]))
# }


def RNN(x):#, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # print(x)
    # x = tf.unstack(x, timesteps, 1)
    # print(tf.shape(x))
    # print("\n\n")
    # x = tf.reshape(x, [batch_size, -1, num_input])

    # Define a lstm cell with tensorflow
    # lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # lstm_cellh1 = rnn.BasicLSTMCell(20, forget_bias=1.0)
    # lstm_cellh2 = rnn.BasicLSTMCell(15, forget_bias=1.0)
    n_cells_layers = [n_h1, n_h2]
    lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_) for n_ in n_cells_layers]
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    # print("state size", multi_lstm_cell .state_size)
    # cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=n_h1), tf.nn.rnn_cell.LSTMCell(num_units=n_h2)])

    # Get lstm cell output
    # outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=x, dtype=tf.float32, scope = "dynamic_rnn")

    fc = tf.contrib.layers.fully_connected(outputs[:,-1], num_classes, activation_fn = None, scope="my_fully_connected")
    # print("+++++++++++++++++++++++++++++ outputs", outputs)
    print("\n\n")
    # # print("---------!!!!!!!!!!!!!!!!!!! states", states[1])
    # print("fc =", fc)
    # print("\n\n")
    # print("\n\n")
    # print("\n\n")
    # print("\n\n")

    # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    # return tf.contrib.layers.fully_connected(states, num_classes)
    return fc

    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    #         tf.matmul(outputs[-1], weights['out']) + biases['out']

# logits = RNN(X, weights, biases)
logits = RNN(X)
prediction = tf.nn.softmax(logits)

# print(tf.shape(logits))
# print("\n\n")
# print("logits", logits)
# print("\n\n")
# print("prediction", prediction)
# print("\n\n")
# print("Y", Y)

# Define loss and optimizer
xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
# print("\n\n")
# print("xentropy", xentropy)
loss_op = tf.reduce_mean(xentropy)
# print("\n\n")
print("loss_op", loss_op)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, name = "train_op")

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

writer = tf.summary.FileWriter('tensorboard/2')

# Initialize the variables (i.e. assign their default value)
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

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128*2
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
