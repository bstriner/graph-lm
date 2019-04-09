import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.rnn_util import lstm
from ...stats import get_bias_ctc


def vae_flat_decoder_attn(latent, vocab_size, params, n, weights_regularizer=None, is_training=True):
    # latent (N, D)
    with tf.variable_scope('decoder'):
        """
        h = slim.fully_connected(
            latent,
            num_outputs=params.decoder_dim,
            scope='projection',
            activation_fn=None,
            weights_regularizer=weights_regularizer
        )
        """
        h = latent
        # h = sequence_norm(h)
        h = slim.batch_norm(h, is_training=is_training)
        h, _ = lstm(
            x=h,
            num_layers=3,
            num_units=params.decoder_dim,
            bidirectional=True
        )
        # h = sequence_norm(h)
        h = slim.batch_norm(h, is_training=is_training)
        """
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
        """
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='decoder_mlp_3',
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.initializers.constant(
                value=get_bias_ctc(average_output_length=params.flat_length, smoothing=params.bias_smoothing),
                verify_shape=True)
        )  # (L,N,V+1)
        return h
