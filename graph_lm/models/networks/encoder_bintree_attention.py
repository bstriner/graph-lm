import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.attn_util import calc_attn_v2
from graph_lm.models.networks.utils.bintree_utils import binary_tree_down, binary_tree_up, stack_tree_v2
from graph_lm.models.networks.utils.rnn_util import lstm
from .embed_sentences import embed_sentences


def encoder_bintree_attn_base(inputs, token_lengths, params, weights_regularizer=None, is_training=True):
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
        """
        if params.batch_norm:
            h = slim.batch_norm(inputs=h, is_training=is_training)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_1',
            weights_regularizer=weights_regularizer
        )
        """
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
        # todo: recurrent attention
        output_tree = binary_tree_down(
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

        flat_attns = stack_tree_v2(attns)
        attn_img = tf.expand_dims(tf.transpose(flat_attns, (1, 0, 2)), axis=3)
        tf.summary.image('encoder_attention', attn_img)

        hs = [tf.matmul(
            attn,  # (n, ol, il)
            tf.transpose(hidden_state, (1, 0, 2)))  # (n, il, d)
            for attn in attns]
        # (n, ol, d)
        hs = [
            tf.concat(cols, axis=-1)
            for cols in zip(hs, output_tree)
        ]
    with tf.variable_scope('encoder_bintree_up'):
        messages_up = binary_tree_up(
            hidden_dim=params.encoder_dim,
            inputs=hs
        )
    with tf.variable_scope('encoder_bintree_down'):
        hs = [
            tf.concat(cols, axis=-1)
            for cols in zip(hs, messages_up)
        ]
        messages_down = binary_tree_down(
            x0=tf.squeeze(hs[0], axis=1),
            hidden_dim=params.encoder_dim,
            depth=params.tree_depth,
            inputs=hs
        )
        hs = [
            tf.concat(cols, axis=-1)
            for cols in zip(hs, messages_down)
        ]
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
