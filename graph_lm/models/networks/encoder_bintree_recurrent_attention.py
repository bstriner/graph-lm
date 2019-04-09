import tensorflow as tf
from tensorflow.contrib import slim

from .embed_sentences import embed_sentences
from .utils.bintree_attn import binary_tree_down_recurrent_attention
from .utils.bintree_utils import concat_layers
from .utils.bintree_utils_v2 import binary_tree_downward_v2, binary_tree_upward_v2
from .utils.rnn_util import lstm


def encoder_bintree_recurrent_attn_base(
        inputs, token_lengths, params,
        weights_regularizer=None, is_training=True):
    """

    :param inputs: (L,N)
    :param token_lengths:
    :return:
    """
    n = tf.shape(inputs)[1]
    with tf.variable_scope('input_lstm'):
        h = inputs
        hidden_state, hidden_state_final = lstm(
            x=h,
            num_layers=3,
            num_units=params.encoder_dim,
            bidirectional=True,
            sequence_lengths=token_lengths
        )
        h = tf.concat(hidden_state_final, axis=-1)  # (layers*directions, N, D)
        h = tf.transpose(h, (1, 0, 2))  # (N,layers*directions,D)
        h = tf.reshape(h, (n, h.shape[1].value * h.shape[2].value))  # (N, layers*directions*D)
        if params.batch_norm:
            h = slim.batch_norm(inputs=h, is_training=is_training)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_1',
            weights_regularizer=weights_regularizer
        )
        if params.batch_norm:
            h = slim.batch_norm(inputs=h, is_training=is_training)
        flat_encoding = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_3',
            weights_regularizer=weights_regularizer
        )  # (N,D)
    with tf.variable_scope('bintree_attention'):
        hs = binary_tree_down_recurrent_attention(
            x0=flat_encoding,
            input_hidden=hidden_state,
            sequence_lengths=token_lengths,
            tree_depth=params.tree_depth,
            params=params)
    with tf.variable_scope('encoder_bintree_up'):
        messages_up = binary_tree_upward_v2(
            hidden_dim=params.encoder_dim,
            inputs=hs
        )
    with tf.variable_scope('encoder_bintree_down'):
        hs = concat_layers(hs, messages_up)
        messages_down = binary_tree_downward_v2(
            hidden_dim=params.encoder_dim,
            inputs=hs
        )
        hs = concat_layers(hs, messages_down)
    return hs


def encoder_bintree_recurrent_attn_vae(
        tokens, token_lengths, vocab_size, params, n, weights_regularizer=None, is_training=True):
    """

    :param tokens: (N,L)
    :param token_lengths: (N,)
    :param vocab_size:
    :param params:
    :param n:
    :param output_length:
    :param weights_regularizer:
    :return:
    """
    with tf.variable_scope('encoder'):
        with tf.variable_scope('embedding'):
            h = tf.transpose(tokens, (1, 0))  # (L,N)
            h = embed_sentences(
                tokens=h,
                token_lengths=token_lengths,
                vocab_size=vocab_size,
                params=params
            )
        with tf.variable_scope('base'):
            hs = encoder_bintree_recurrent_attn_base(
                inputs=h,
                token_lengths=token_lengths,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training
            )
        with tf.variable_scope('output_projection'):
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h1',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h2',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            mu = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.latent_dim,
                    activation_fn=None,
                    scope='encoder_mlp_mu',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            logsigma = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.latent_dim,
                    activation_fn=None,
                    scope='encoder_mlp_logsigma',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            return mu, logsigma


def encoder_bintree_recurrent_attn_aae(
        tokens, token_lengths, noise, vocab_size, params, n, weights_regularizer=None, is_training=True):
    """

    :param tokens: (N,L)
    :param token_lengths: (N,)
    :param vocab_size:
    :param params:
    :param n:
    :param output_length:
    :param weights_regularizer:
    :return:
    """
    with tf.variable_scope('encoder'):
        with tf.variable_scope('embedding'):
            h = tf.transpose(tokens, (1, 0))  # (L,N)
            h = embed_sentences(
                tokens=h,
                token_lengths=token_lengths,
                vocab_size=vocab_size,
                params=params
            )
            noise_t = tf.transpose(noise, (1, 0, 2))
            h = tf.concat([h, noise_t], axis=-1)
        with tf.variable_scope('base'):
            hs = encoder_bintree_recurrent_attn_base(
                inputs=h,
                token_lengths=token_lengths,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training
            )
        with tf.variable_scope('output_projection'):
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h1',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h2',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            return hs
