import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION, CUDNN_RNN_UNIDIRECTION, CudnnLSTM


def make_initial_states(batch_size, num_layers, dim, bidirectional=False):
    num_dirs = 2 if bidirectional else 1
    initial_c = tf.get_variable(
        name='initial_c',
        dtype=tf.float32,
        initializer=tf.initializers.zeros,
        shape=(num_layers * num_dirs, 1, dim))
    initial_h = tf.get_variable(
        name='initial_h',
        dtype=tf.float32,
        initializer=tf.initializers.zeros,
        shape=(num_layers * num_dirs, 1, dim))
    initial_h, initial_c = [
        tf.tile(s, [1, batch_size, 1])
        for s in [initial_h, initial_c]
    ]
    return initial_h, initial_c


def lstm(x, num_layers, num_units, sequence_lengths=None, dropout=0., bidirectional=False):
    batch_size = tf.shape(x)[1]
    initial_states = make_initial_states(
        batch_size=batch_size,
        num_layers=num_layers,
        dim=num_units,
        bidirectional=bidirectional)
    lstm = CudnnLSTM(
        num_layers=num_layers,
        num_units=num_units,
        direction=CUDNN_RNN_BIDIRECTION if bidirectional else CUDNN_RNN_UNIDIRECTION,
        dropout=dropout
    )
    ret = lstm(x, sequence_lengths=sequence_lengths, initial_state=initial_states)
    return ret
