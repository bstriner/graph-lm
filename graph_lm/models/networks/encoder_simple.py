import tensorflow as tf
from tensorflow.contrib import slim

from ..rnn_util import lstm


def vae_flat_encoder_simple(
        tokens, token_lengths,
        vocab_size, params, n, weights_regularizer=None):
    """

    :param tokens: (N,L)
    :param token_lengths: (N,)
    :param vocab_size:
    :param params:
    :param n:
    :param weights_regularizer:
    :return:
    """
    L = tf.shape(tokens)[1]
    with tf.variable_scope('encoder'):
        embeddings = tf.get_variable(
            dtype=tf.float32,
            name="embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        embedded_tokens = tf.nn.embedding_lookup(params=embeddings, ids=tf.transpose(tokens, (1, 0)))  # (L, N, D)
        ls = tf.linspace(
            start=tf.constant(0, dtype=tf.float32),
            stop=tf.constant(1, dtype=tf.float32),
            num=L)  # (L,)
        ls = tf.tile(tf.expand_dims(ls, 1), [1, n])  # (L,N)
        ls = ls * tf.cast(L, dtype=tf.float32) / tf.cast(tf.expand_dims(token_lengths, 0), dtype=tf.float32)
        ls = tf.expand_dims(ls, 2)  # ( L,N,1)
        h = tf.concat([embedded_tokens, ls], axis=-1)
        h, _ = lstm(
            x=h,
            num_layers=3,
            num_units=params.encoder_dim,
            bidirectional=True,
            sequence_lengths=token_lengths
        )
        mu = slim.fully_connected(
            inputs=h,
            num_outputs=params.latent_dim,
            activation_fn=None,
            scope='encoder_mlp_mu',
            weights_regularizer=weights_regularizer
        )
        logsigma = slim.fully_connected(
            inputs=h,
            num_outputs=params.latent_dim,
            activation_fn=None,
            scope='encoder_mlp_logsigma',
            weights_regularizer=weights_regularizer
        )
        return mu, logsigma
