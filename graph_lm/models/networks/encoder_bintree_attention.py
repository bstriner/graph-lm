import tensorflow as tf
from tensorflow.contrib import slim

from ..attn_util import calc_attn_v2
from ..rnn_util import lstm
from .bintree_utils import binary_tree_resnet

from .embed_sentences import embed_sentences


def encoder_bintree_attn_base(inputs, token_lengths, params, weights_regularizer=None, is_training=True):
    """

    :param inputs: (L,N)
    :param token_lengths:
    :return:
    """
    n = tf.shape(inputs)[1]
    with tf.variable_scope('step_1_base'):
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
    with tf.variable_scope('step_2'):
        output_tree = binary_tree_resnet(
            x0=flat_encoding,
            hidden_dim=params.encoder_dim,
            depth=params.tree_depth
        )

        output_projs = [slim.fully_connected(
            inputs=enc,
            num_outputs=params.attention_dim,
            activation_fn=None,
            scope='encoder_mlp_encoding',
            weights_regularizer=weights_regularizer,
            reuse=i > 0
        )
            for i, enc in enumerate(output_tree)
        ]
        input_proj = slim.fully_connected(
            inputs=hidden_state,
            num_outputs=params.attention_dim,
            activation_fn=None,
            scope='encoder_input_proj',
            weights_regularizer=weights_regularizer
        )  # (O,N,D)
        attns = [
            calc_attn_v2(
                output_proj, input_proj, token_lengths,
                a_transpose=False,
                b_transpose=True
            )
            for output_proj in output_projs
        ]  # (n, ol, il)
        # tf.summary.image('encoder_attention', tf.expand_dims(attn, 3))
        input_aligneds = [tf.matmul(
            attn,  # (n, ol, il)
            tf.transpose(hidden_state, (1, 0, 2)))  # (n, il, d)
            for attn in attns]
        # (n, ol, d)
        input_aligneds = [
            tf.concat([i, j], axis=-1)
            for i, j in zip(input_aligneds, output_tree)
        ]
    with tf.variable_scope('encoder_output'):
        hs = binary_tree_resnet(
            x0=tf.squeeze(input_aligneds[0], axis=1),
            hidden_dim=params.encoder_dim,
            depth=params.tree_depth,
            inputs=input_aligneds
        )
        return hs


def encoder_bintree_attn(tokens, token_lengths, vocab_size, params, n, weights_regularizer=None
                         , is_training=True):
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
            hs = encoder_bintree_attn_base(
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
                    activation_fn=None,
                    scope='encoder_mlp_output_projection_h',
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
