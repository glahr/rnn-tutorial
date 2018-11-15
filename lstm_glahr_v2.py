"""Training with long time series.

    Supplementary code for:
    D. Hafner and C. Igel. "Signal Processing with Recurrent Neural Networks in TensorFlow"
    """

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
nest = tf.contrib.framework.nest

def lstm_func(d, m, n, lr):

    # Placeholders
    inputs = tf.placeholder(tf.float32, [None, None, d])
    targets = tf.placeholder(tf.float32, [None, None, m])

    # Network architecture
    n_cells_layers = [20, 15]
    cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_) for n_ in n_cells_layers]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    zero_state = cell.zero_state(n, tf.float32)

    state =  nest.map_structure(lambda tensor: tf.Variable(tensor, trainable=False), zero_state)

    # RNN
    # rnn_output, new_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=state, dtype=tf.float32)
    rnn_output, new_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    # State update
    update_state = nest.map_structure(tf.assign, state, new_state)
    update_state = nest.flatten(update_state)
    reset_state  = nest.map_structure(tf.assign, state, zero_state)
    reset_state  = nest.flatten(reset_state)

    # print(update_state, "\n")
    # print(reset_state, "\n")
    # print("\n\n")
    tf.summary.scalar('update_state', update_state)
    # tf.summary.histogram('reset_state', reset_state)
    # tf.summary.histogram('rnn_output', rnn_output)
    # merged_summary = tf.summary.merge_all()

    with tf.control_dependencies(update_state):  # Update_state is already a list
        rnn_output = tf.identity(rnn_output)

    rnn_output_flat = tf.reshape(rnn_output, [-1, n_cells_layers[-1]])
    prediction_flat = tf.layers.dense(rnn_output_flat, m, activation=None)
    targets_flat = tf.reshape(targets, [-1, m])
    prediction  = tf.reshape(prediction_flat, [-1, tf.shape(inputs)[1], m])
    # Error function and optimizer
    loss = tf.losses.mean_squared_error(targets_flat, prediction_flat)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    return inputs, targets, new_state, reset_state, prediction, loss, train_step


def main():
    # Parameters
    gap = 5  # Time steps to predict into the future
    T = 600  # Length of training time series
    # N = [32]  # Size of recurrent neural network
    n = 1  # Number of training sequences
    n_test = 1  # Number of test sequences
    m = 1  # Output dimension
    d = 1  # Input dimension
    epochs = 200  # Maximum number of epochs
    lr = 0.01  # Learning rate

    # Load and arrange data
    raw_data = np.genfromtxt('data/data_adjust-1.csv',delimiter = ",")
    train_X = raw_data[0:T,9]
    train_X = train_X.copy()
    train_Y = raw_data[gap:T+gap,9]
    train_Y = train_Y.copy()
    test_X = raw_data[T:-gap,9]
    test_X = test_X.copy()
    test_Y = raw_data[T+gap:,9]
    test_Y = test_Y.copy()

    train_X.resize(n, train_X.size, d)
    train_Y.resize(n, train_Y.size, m)
    test_X.resize(n_test, test_X.size, d)
    test_Y.resize(n_test, test_Y.size, m)
    time = np.arange(train_X.size)*0.012

    inputs, targets, new_state, reset_state, prediction, loss, train_step = lstm_func(d, m, n, lr)

    writer = tf.summary.FileWriter('tensorboard/1')

    # Create session and initialize variables
    with tf.Session() as sess:
        writer.add_graph(sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.graph.finalize()  # Graph is read-only after this statement.
        # Do the learning
        for i in range(epochs):
            sess.run(reset_state)  # Reset  at beginning of each time series
            chunk_size = 50
            for chunk_start in range(0, T, chunk_size):
                sess.run(train_step, feed_dict={
                    inputs: train_X[:, chunk_start: chunk_start + chunk_size],
                    targets: train_Y[:, chunk_start: chunk_start + chunk_size]})
            if (i+1)%10==0:
                sess.run(reset_state)
                temp_loss = sess.run(loss, feed_dict={inputs: train_X, targets: train_Y})
                # s = sess.run(merged_summary, feed_dict={inputs: train_X, targets: train_Y})
                # writer.add_summary(s, i)
                print(i+1, ' loss =', temp_loss)

        # # Visualize modelling of training data
        # # sess.run(reset_state)
        # model, final_state = sess.run([prediction, new_state], feed_dict={inputs: train_X})
        # plt.rc('font', size=14)
        # # plt.plot(train_X[0,:,0], label='input', color='lightgray', linestyle='--')
        # plt.plot(time, train_Y[0,:,0], label='target', linestyle='-', linewidth=3)
        # plt.plot(time, model[0,:,0], label='model', linestyle='-', linewidth=3)
        # plt.legend(loc=1)
        # plt.xlabel('time [t]')
        # plt.ylabel('signal')
        # plt.title('data presented in one batch')
        # # plt.savefig('lorenzTrainChunk.pdf')
        # plt.show()

        sess.run(reset_state)
        concatenated = []
        for chunk_start in range (0, T, chunk_size):
            model, _ = sess.run([prediction, new_state], feed_dict={
                           inputs:
                           train_X[:, chunk_start: chunk_start + chunk_size]})
            concatenated.append(model)
        model = np.stack(concatenated, axis=0)
        model = np.reshape(model, model.size)
        plt.plot(time, train_Y[0,:,0], label='target', linestyle='-', linewidth=3)
        plt.plot(time, model, label='model', linestyle='-', linewidth=3)
        plt.legend(loc=1)
        plt.xlabel('time [t]')
        plt.ylabel('signal')
        plt.title('data presented in chunks')
        # plt.savefig('lorenzTrainChunkEvalChunk.pdf')
        plt.show()


if __name__ == "__main__":
    main()
