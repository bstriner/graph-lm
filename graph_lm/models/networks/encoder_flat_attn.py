import tensorflow as tf
from tensorflow.contrib import slim

from ..attn_util import calc_attn_v2
from ..rnn_util import lstm


def vae_flat_encoder_attn(tokens, token_lengths, vocab_size, params, n, output_length, weights_regularizer=None
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
    L = tf.shape(tokens)[1]
    with tf.variable_scope('encoder'):
        with tf.variable_scope('step_1'):
            h = tf.transpose(tokens, (1, 0))  # (L,N)
            embeddings = tf.get_variable(
                dtype=tf.float32,
                name="embeddings",
                shape=[vocab_size, params.encoder_dim],
                initializer=tf.initializers.truncated_normal(
                    stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
            h = tf.nn.embedding_lookup(embeddings, h)  # (L, N, D)
            ls = tf.linspace(
                start=tf.constant(0, dtype=tf.float32),
                stop=tf.constant(1, dtype=tf.float32),
                num=L)  # (L,)
            ls = tf.tile(tf.expand_dims(ls, 1), [1, n])  # (L,N)
            ls = ls * tf.cast(L, dtype=tf.float32) / tf.cast(tf.expand_dims(token_lengths, 0), dtype=tf.float32)
            ls = tf.expand_dims(ls, 2)  # ( L,N,1)
            h = tf.concat([h, ls], axis=-1)
            hidden_state, hidden_state_final = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True,
                sequence_lengths=token_lengths
            )
            h = tf.concat(hidden_state_final, axis=-1)  # (layers*directions, N, D)
            h = tf.transpose(h, (1, 0, 2))  # (N,layers*directions,D)
            h = tf.reshape(h, (n, h.shape[1].value * h.shape[2].value))  # (N, layers*directions*D)
            h = slim.batch_norm(inputs=h, is_training=True)
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_1',
                weights_regularizer=weights_regularizer
            )
            h = slim.batch_norm(inputs=h, is_training=True)
            """
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_2',
                weights_regularizer=weights_regularizer
            )
            """
            flat_encoding = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_3',
                weights_regularizer=weights_regularizer
            )  # (N,D)
        with tf.variable_scope('step_2'):
            h = tf.expand_dims(flat_encoding, axis=0)  # (1, N, D)
            h = tf.tile(h, (output_length, 1, 1))  # (O,N,D)
            ls = tf.linspace(start=-1., stop=1., num=params.flat_length)  # (O,)
            ls = tf.tile(tf.expand_dims(tf.expand_dims(ls, 1), 2), (1, n, 1))  # (O,N,1)
            h = tf.concat([h, ls], axis=2)
            output_hidden, _ = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True
            )  # (O, N, D)
            # output_hidden = sequence_norm(output_hidden)
            output_hidden = slim.batch_norm(inputs=output_hidden, is_training=is_training)
        with tf.variable_scope('encoder_attn'):
            output_proj = slim.fully_connected(
                inputs=output_hidden,
                num_outputs=params.attention_dim,
                activation_fn=None,
                scope='encoder_output_proj',
                weights_regularizer=weights_regularizer
            )  # (O,N,D)
            input_proj = slim.fully_connected(
                inputs=hidden_state,
                num_outputs=params.attention_dim,
                activation_fn=None,
                scope='encoder_input_proj',
                weights_regularizer=weights_regularizer
            )  # (O,N,D)
            attn = calc_attn_v2(output_proj, input_proj, token_lengths)  # (n, ol, il)
            tf.summary.image('encoder_attention', tf.expand_dims(attn, 3))
            input_aligned = tf.matmul(
                attn,  # (n, ol, il)
                tf.transpose(hidden_state, (1, 0, 2))  # (n, il, d)
            )  # (n, ol, d)
            h = tf.concat([tf.transpose(input_aligned, (1, 0, 2)), output_hidden], axis=-1)
        with tf.variable_scope('encoder_output'):
            # h = sequence_norm(h)
            h = slim.batch_norm(h, is_training=is_training)
            h, _ = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True
            )  # (O, N, D)
            """
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=None,
                scope='encoder_mlp_out_1',
                weights_regularizer=weights_regularizer
            )
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=None,
                scope='encoder_mlp_out_2',
                weights_regularizer=weights_regularizer
            )
            """
            # h = sequence_norm(h)
            h = slim.batch_norm(h, is_training=is_training)
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
