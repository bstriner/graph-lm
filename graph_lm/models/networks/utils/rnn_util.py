import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION, CUDNN_RNN_UNIDIRECTION, \
    CudnnLSTM


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


def lstm(x, num_layers, num_units, scope='lstm', sequence_lengths=None, dropout=0., bidirectional=False):
    with tf.variable_scope(scope):
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


def sequence_norm(x, epsilon=1e-5):
    """

    :param x: (L, N, D)
    :return:
    """
    ex = tf.reduce_mean(x, axis=0, keepdims=True)  # (1, N, D)
    h = x - ex
    sx = tf.sqrt(tf.reduce_mean(tf.square(h), axis=0, keepdims=True))  # (1,N,D)
    h = h / tf.maximum(sx, epsilon)
    return h


def linspace_feature(L, N, start=-1., stop=1.):
    h = tf.linspace(start=start, stop=stop, num=L)  # (L,)
    h = tf.expand_dims(h, axis=1)  # (L,1)
    h = tf.tile(h, (1, N))
    h = tf.expand_dims(h, axis=2)  # (L,N,1)
    return h

def linspace_scaled_feature(L, N, sequence_length):
    Lf = tf.cast(L, dtype=tf.float32)
    h = linspace_feature(L, N, start=0., stop=1.)
    lengths = tf.expand_dims(tf.expand_dims(tf.cast(sequence_length, dtype=tf.float32), axis=0), axis=2) # (1,N,1)
    h = h * Lf / lengths
    return h
