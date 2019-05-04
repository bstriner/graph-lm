import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.rnn_util import lstm
from .utils.rnn_util import linspace_feature, linspace_scaled_feature
from ...models import AAE_STOCH, AE


def encoder_flat(tokens, token_lengths, vocab_size, params, n, weights_regularizer=None, is_training=True):
    with tf.variable_scope('encoder'):
        N = tf.shape(tokens)[0]
        L = tf.shape(tokens)[1]
        h = tf.transpose(tokens, (1, 0))  # (L,N)
        embeddings = tf.get_variable(
            dtype=tf.float32,
            name="embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        inputs = [
            tf.nn.embedding_lookup(embeddings, h),  # (L, N, D)
            linspace_feature(L=L, N=N),
            linspace_scaled_feature(L=L, N=N, sequence_length=token_lengths)]
        if params.model_mode == AAE_STOCH:
            noise = tf.random_normal(
                shape=(L, N, params.noise_dim),
                dtype=tf.float32
            )
            inputs.append(noise)
        h = tf.concat(inputs, axis=-1)
        _, h = lstm(
            x=h,
            num_layers=params.encoder_layers,
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
        if params.batch_norm:
            h = slim.batch_norm(h, is_training=is_training)
        """
        for i in range(params.encoder_layers):
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_{}'.format(i),
                weights_regularizer=weights_regularizer
            )
            if params.batch_norm:
                h = slim.batch_norm(h, is_training=is_training)
        """
        if params.model_mode == AAE_STOCH or params.model_mode == AE:
            encoding = slim.fully_connected(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='encoder_mlp_encoding',
                weights_regularizer=weights_regularizer
            )
            return encoding, None
        else:
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
