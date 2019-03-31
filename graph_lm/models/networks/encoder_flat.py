import tensorflow as tf
from tensorflow.contrib import slim

from ..rnn_util import lstm


def vae_flat_encoder(tokens, token_lengths, vocab_size, params, n, weights_regularizer=None):
    with tf.variable_scope('encoder'):
        h = tf.transpose(tokens, (1, 0))  # (L,N)
        embeddings = tf.get_variable(
            dtype=tf.float32,
            name="embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        h = tf.nn.embedding_lookup(embeddings, h)  # (L, N, D)
        _, h = lstm(
            x=h,
            num_layers=3,
            num_units=params.encoder_dim,
            bidirectional=True,
            sequence_lengths=token_lengths
        )
        print("h1: {}".format(h))
        # h = h[1]  # [-2:, :, :]  # (2, N, D)
        h = tf.concat(h, axis=-1)
        print("h2: {}".format(h))
        h = tf.transpose(h, (1, 0, 2))  # (N,2,D)
        print("h3: {}".format(h))
        h = tf.reshape(h, (n, h.shape[1].value * h.shape[2].value))  # (N, 2D)
        print("h4: {}".format(h))
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_2',
            weights_regularizer=weights_regularizer
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
