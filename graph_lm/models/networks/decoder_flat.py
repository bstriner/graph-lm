import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.rnn_util import lstm


def vae_flat_decoder(latent, vocab_size, params, n, weights_regularizer=None):
    # latent (N, D)
    with tf.variable_scope('decoder'):
        depth = params.tree_depth
        assert depth >= 0
        h = slim.fully_connected(
            latent,
            num_outputs=params.decoder_dim,
            scope='projection',
            activation_fn=None,
            weights_regularizer=weights_regularizer
        )
        h = tf.expand_dims(h, axis=0)  # (1, N, D)
        h = tf.tile(h, (params.flat_length, 1, 1))  # (L,N,D)
        h, _ = lstm(
            x=h,
            num_layers=3,
            num_units=params.decoder_dim,
            bidirectional=True
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='decoder_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='decoder_mlp_2',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='decoder_mlp_3',
            weights_regularizer=weights_regularizer
        )  # (L,N,V+1)
        return h
