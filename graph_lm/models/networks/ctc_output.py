import math
import tensorflow as tf
from tensorflow.contrib import slim

from ...stats import get_bias_ctc


def calc_output(x, vocab_size, params, weights_regularizer=None, reuse=False, is_training=True):
    # X: (N,*, D)
    # Y: (N,*, V)
    with tf.variable_scope('output_mlp', reuse=reuse):
        h = x
        if params.batch_norm:
            h = slim.batch_norm(h, is_training=is_training)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2',
            weights_regularizer=weights_regularizer
        )
        if params.batch_norm:
            h = slim.batch_norm(h, is_training=is_training)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.initializers.constant(get_bias_ctc(
                average_output_length=math.pow(2, params.tree_depth + 1) - 1,
                smoothing=0.05
            ))
        )
    return h
