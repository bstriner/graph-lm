import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.rnn_util import lstm
from .utils.rnn_util import linspace_feature, linspace_scaled_feature


def decoder_flat(latent, vocab_size, params, n, weights_regularizer=None):
    # latent (N, D)
    N = tf.shape(latent)[0]
    L = params.flat_length
    with tf.variable_scope('decoder'):
        h = tf.expand_dims(latent, axis=0)  # (1, N, D)
        h = tf.tile(h, (L, 1, 1))  # (L,N,D)
        h = tf.concat([h, linspace_feature(N=N, L=L)], axis=-1)
        h, _ = lstm(
            x=h,
            num_layers=params.decoder_layers,
            num_units=params.decoder_dim,
            bidirectional=True
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='decoder_mlp_output',
            weights_regularizer=weights_regularizer
        )  # (L,N,V+1)
        return h
