import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.dag_utils import message_passing
from ...stats import get_bias


def decoder_dag_supervised(latent, dag, dag_bw,dag_feats, sequence_length, vocab_size, params, weights_regularizer=None,
                           is_training=True):
    # latent (N, L, D)
    N = tf.shape(latent)[0]
    L = tf.shape(latent)[1]
    with tf.variable_scope('decoder'):
        #h = tf.concat([latent, tf.expand_dims(h_linspace, -1)], axis=-1)
        with tf.variable_scope("upward"):
            h = message_passing(
                latent=dag_feats,
                dag_bw=dag_bw,
                params=params,
                dim=params.decoder_dim,
                hidden_depth=params.decoder_layers,
                weights_regularizer=weights_regularizer
            )
        h = tf.concat([h, latent], axis=-1)
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
