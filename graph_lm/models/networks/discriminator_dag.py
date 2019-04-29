import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.dag_utils import message_passing
from ...stats import get_bias


def discriminator_dag(latent, dag, dag_bw, sequence_length, params, weights_regularizer=None,
                      is_training=True):
    # latent (N, L, D)
    N = tf.shape(latent)[0]
    L = tf.shape(latent)[1]
    with tf.variable_scope('discriminator'):
        h_linspace = tf.linspace(start=0, stop=L, num=L)
        h_linspace = tf.tile(tf.expand_dims(h_linspace, 0), [N, 1])
        h_linspace = h_linspace / tf.expand_dims(sequence_length, axis=1)
        h = tf.concat([latent, tf.expand_dims(h_linspace, -1)], axis=-1)
        with tf.variable_scope("upward"):
            h = message_passing(
                latent=h,
                dag_bw=dag_bw,
                params=params,
                dim=params.discriminator_dim,
                hidden_depth=params.discriminator_layers,
                weights_regularizer=weights_regularizer
            )
        with tf.variable_scope("downward"):
            h = message_passing(
                latent=h,
                dag_bw=dag,
                params=params,
                dim=params.discriminator_dim,
                hidden_depth=params.discriminator_layers,
                weights_regularizer=weights_regularizer
            )
        with tf.variable_scope('output_mlp'):
            for i in range(params.discriminator_layers):
                h = slim.fully_connected(
                    inputs=h,
                    activation_fn=tf.nn.leaky_relu,
                    weights_regularizer=weights_regularizer,
                    num_outputs=params.discriminator_dim,
                    scope='discriminator_output_{}'.format(i)
                )
            logits = slim.fully_connected(
                inputs=h,
                num_outputs=1,
                activation_fn=None,
                scope='discriminator_output_logits',
                weights_regularizer=weights_regularizer,
                biases_initializer=tf.initializers.constant(
                    value=get_bias(smoothing=params.bias_smoothing),
                    verify_shape=True)
            )  # (N,L,1)
        return logits
