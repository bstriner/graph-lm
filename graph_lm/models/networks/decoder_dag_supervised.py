import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.dag_utils import message_passing
from ...stats import get_bias
from .utils.rnn_util import lstm

def decoder_dag_supervised(latent, dag, dag_bw, sequence_length, dag_feats, vocab_size, params, weights_regularizer=None,
                           is_training=True):
    # latent (N, L, D)
    with tf.variable_scope('decoder'):
        h = tf.concat([latent, dag_feats], axis=-1)
        with tf.variable_scope("upward"):
            h = message_passing(
                latent=h,
                dag_bw=dag_bw,
                params=params,
                dim=params.decoder_dim,
                hidden_depth=params.decoder_layers,
                weights_regularizer=weights_regularizer
            )
        with tf.variable_scope("downward"):
            h = message_passing(
                latent=h,
                dag_bw=dag,
                params=params,
                dim=params.decoder_dim,
                hidden_depth=params.decoder_layers,
                weights_regularizer=weights_regularizer
            )
        with tf.variable_scope('output_mlp'):
            if params.lstm_output:
                h, _ = lstm(
                    x=h,
                    num_units=params.decoder_dim,
                    bidirectional=True,
                    num_layers=params.decoder_layers,
                    sequence_lengths=sequence_length
                )
            else:
                for i in range(params.decoder_layers):
                    h = slim.fully_connected(
                        inputs=h,
                        activation_fn=tf.nn.leaky_relu,
                        weights_regularizer=weights_regularizer,
                        num_outputs=params.decoder_dim,
                        scope='decoder_output_{}'.format(i)
                    )
            logits = slim.fully_connected(
                inputs=h,
                num_outputs=vocab_size,
                activation_fn=None,
                scope='decoder_output_logits',
                weights_regularizer=weights_regularizer,
                #biases_initializer=tf.initializers.constant(
                #    value=get_bias(smoothing=params.bias_smoothing),
                #    verify_shape=True)
            )  # (N,L,V)
        return logits
