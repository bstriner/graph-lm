import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.dag_utils import make_dag, message_passing
from graph_lm.models.networks.utils.rnn_util import lstm
from ...stats import get_bias


def process_latent(latent, sequence_lengths, params):
    latent, _ = lstm(
        x=tf.transpose(latent, (1, 0, 2)),
        num_layers=2,
        num_units=params.decoder_dim,
        bidirectional=True,
        sequence_lengths=sequence_lengths
    )
    latent = tf.transpose(latent, (1, 0, 2))
    return latent


def vae_decoder_dag(latent, sequence_lengths, vocab_size, params, n, weights_regularizer=None, is_training=True):
    # latent (N, L, D)
    with tf.variable_scope('decoder'):
        latent_processed = process_latent(latent=latent, sequence_lengths=sequence_lengths, params=params)

        dag, penalty = make_dag(
            latent=latent_processed,
            sequence_lengths=sequence_lengths,
            params=params
        )

        dag_bw = tf.transpose(dag, (0, 2, 1))
        hidden = message_passing(
            latent=latent,
            dag_bw=dag,
            params=params
        )
        hidden = message_passing(
            latent=hidden,
            dag_bw=dag_bw,
            params=params
        )
        # hidden (N, L, Dlatent+decoder_dim)

        h = slim.fully_connected(
            inputs=hidden,
            activation_fn=tf.nn.leaky_relu,
            num_outputs=params.decoder_dim,
            scope='output_1'
        )
        logits = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size,
            activation_fn=None,
            scope='output_2',
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.initializers.constant(
                value=get_bias(smoothing=params.bias_smoothing),
                verify_shape=True)
        )  # (N,L,V)
        return logits, penalty
