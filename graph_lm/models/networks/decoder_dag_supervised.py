import tensorflow as tf
from tensorflow.contrib import slim

from .dag_utils import make_dag, message_passing
from ..rnn_util import lstm
from ...stats import get_bias


def vae_decoder_dag_supervised(latent, dag, dag_bw, vocab_size, params, tags, tag_size, weights_regularizer=None, is_training=True):
    # latent (N, L, D)
    with tf.variable_scope('decoder'):
        tag_embeddings = tf.get_variable(
            dtype=tf.float32,
            name="tag_embeddings",
            shape=[tag_size, params.decoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        h_tags = tf.nn.embedding_lookup(params=tag_embeddings, ids=tags)
        h = tf.concat([latent, h_tags], axis=-1)
        with tf.variable_scope("forward"):
            h = message_passing(
                latent=h,
                dag_bw=dag_bw,
                params=params
            )
        with tf.variable_scope("backward"):
            h = message_passing(
                latent=h,
                dag_bw=dag,
                params=params
            )
        with tf.variable_scope('output_mlp'):
            h = slim.fully_connected(
                inputs=h,
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
        return logits
