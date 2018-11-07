"""Training with long time series.

    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
nest = tf.contrib.framework.nest

# Parameters
gap = 5  # Time steps to predict into the future
T = 500  # Length of training time series
N = 32  # Size of recurrent neural network
n = 1  # Number of training sequences
n_test = 1  # Number of test sequences
m = 1  # Output dimension
d = 1  # Input dimension
epochs = 200  # Maximum number of epochs
lr = 0.01  # Learning rate

# Load and arrange data
raw_data = np.genfromtxt('data/lorenz1000.dt')
train_X = raw_data[0:T]
train_Y = raw_data[0+gap:T+gap]
test_X = raw_data[T:-gap]
test_Y = raw_data[T+gap:]
train_X.resize(n, train_X.size, d)
train_Y.resize(n, train_Y.size, m)
test_X.resize(n_test, test_X.size, d)
test_Y.resize(n_test, test_Y.size, m)

# Placeholders
inputs = tf.placeholder(tf.float32, [None, None, d])
targets = tf.placeholder(tf.float32, [None, None, m])

# Network architecture
# cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
# cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
# cell = tf.nn.rnn_cell.BasicLSTMCell(N)
cell = tf.nn.rnn_cell.LSTMCell(N)
# num_layers = 2
n_cell_h1 = 20
n_cell_h2 = 15
cell_h1 = tf.nn.rnn_cell.LSTMCell(n_cell_h1)
print("n_cell_h1.get_config = " + str(cell_h1.state_size))
cell_h2 = tf.nn.rnn_cell.LSTMCell(n_cell_h2)
# print(([cell]*num_layers).state_size)
# cell = tf.nn.rnn_cell.MultiRNNCell([cell for n in range(num_layers)], state_is_tuple=True)
# cell = tf.nn.rnn_cell.MultiRNNCell([cell_h1, cell_h2])
print("cell.state_size = " + str(cell.state_size))
print("cell.get_config = " + str(cell.get_config))

# A state with all variables set to zero
zero_state = cell.zero_state(n, tf.float32)
# State
state =  nest.map_structure(lambda tensor: tf.Variable(tensor, trainable=False), zero_state)
print("state = " + str(state))

# RNN
rnn_output, new_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state, dtype=tf.float32)
print("rnn_output.shape = " + str(rnn_output.shape))

print("\n\n")

# State update
update_state = nest.map_structure(tf.assign, state, new_state)
update_state = nest.flatten(update_state)
reset_state  = nest.map_structure(tf.assign, state, zero_state)
reset_state  = nest.flatten(reset_state)

with tf.control_dependencies(update_state):  # Update_state is already a list
    rnn_output = tf.identity(rnn_output)

# Note the following reshaping:
#   We want a prediction for every time step.
#   Weights of fully connected layer should be the same (shared) for every time step.
#   This is achieved by flattening the first two dimensions.
#   Now all time steps look the same as individual inputs in a batch fed into a feed-forward network.
# rnn_output_flat = tf.reshape(rnn_output, [-1, n_cell_h1, n_cell_h2])
rnn_output_flat = tf.reshape(rnn_output, [-1, N])
# print(rnn_output_flat.get_shape())
prediction_flat = tf.layers.dense(rnn_output_flat, m, activation=None)

targets_flat = tf.reshape(targets, [-1, m])
prediction  = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])
#my changes

# print(rnn_output_flat.get_shape())
# print(prediction_flat.get_shape())

# Error function and optimizer
loss = tf.losses.mean_squared_error(targets_flat, prediction_flat)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Create session and initialize variables
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # Graph is read-only after this statement.

    # Do the learning
    for i in range(epochs):
        sess.run(reset_state)  # Reset  at beginning of each time series
        chunk_size = 50
        for chunk_start in range(0, T, chunk_size):
            # print((train_X[:, chunk_start: chunk_start + chunk_size]).shape)
            # print((train_Y[:, chunk_start: chunk_start + chunk_size]).shape)
            # print(train_step)
            sess.run(train_step, feed_dict={
                inputs: train_X[:, chunk_start: chunk_start + chunk_size],
                targets: train_Y[:, chunk_start: chunk_start + chunk_size]})
        if (i+1)%10==0:
            sess.run(reset_state)
            temp_loss = sess.run(loss, feed_dict={inputs: train_X, targets: train_Y})
            print(i+1, ' loss =', temp_loss)

    # Visualize modelling of training data
    sess.run(reset_state)
    model, final_state = sess.run([prediction, new_state], feed_dict={inputs: train_X})
    plt.rc('font', size=14)
    # plt.plot(train_X[0,:,0], label='input', color='lightgray', linestyle='--')
    plt.plot(train_Y[0,:,0], label='target', linestyle='-', linewidth=3)
    plt.plot(model[0,:,0], label='model', linestyle='-', linewidth=3)
    plt.legend(loc=1)
    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.title('data presented in one batch')
    plt.savefig('lorenzTrainChunk.pdf')
    plt.show()

    sess.run(reset_state)
    concatenated = []
    for chunk_start in range (0, T, chunk_size):
        model, _ = sess.run([prediction, new_state], feed_dict={
                       inputs:
                       train_X[:, chunk_start: chunk_start + chunk_size]})
        concatenated.append(model)
    model = np.stack(concatenated, axis=0)
    model = np.reshape(model, model.size)
    plt.plot(train_Y[0,:,0], label='target', linestyle='-', linewidth=3)
    plt.plot(model, label='model', linestyle='-', linewidth=3)
    plt.legend(loc=1)
    plt.xlabel('time [t]')
    plt.ylabel('signal')
    plt.title('data presented in chunks')
    plt.savefig('lorenzTrainChunkEvalChunk.pdf')
    plt.show()
