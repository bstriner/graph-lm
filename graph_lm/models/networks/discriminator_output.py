import math
import tensorflow as tf
from tensorflow.contrib import slim

from ...stats import get_bias_ctc
from ...sn import sn_fully_connected

def discriminator_output(x, params, weights_regularizer=None, reuse=False, is_training=True):
    # X: (N,*, D)
    # Y: (N,*, V)
    with tf.variable_scope('output_discriminator', reuse=reuse):
        h = x
        for i in range(params.discriminator_layers):
            h = sn_fully_connected(
                inputs=h,
                num_outputs=params.discriminator_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='output_mlp_{}'.format(i),
                weights_regularizer=weights_regularizer
            )
        h = sn_fully_connected(
            inputs=h,
            num_outputs=1,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer)
    return h
