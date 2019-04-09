import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.dag_utils import message_passing


def encoder_dag_gan(dag, noise, dag_bw, text, tags, vocab_size, tags_size, params, weights_regularizer=None):
    with tf.variable_scope("encoder"):
        text_embeddings = tf.get_variable(
            dtype=tf.float32,
            name="text_embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        tag_embeddings = tf.get_variable(
            dtype=tf.float32,
            name="tag_embeddings",
            shape=[tags_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        h_text = tf.nn.embedding_lookup(params=text_embeddings, ids=text)  # (L, N, D)
        h_tags = tf.nn.embedding_lookup(params=tag_embeddings, ids=tags)  # (L, N, D)
        h = tf.concat([h_text, h_tags, noise], axis=-1)
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
        encoding = slim.fully_connected(
            inputs=h,
            num_outputs=params.latent_dim,
            activation_fn=None,
            scope='encoder_mlp_output',
            weights_regularizer=weights_regularizer
        )
        return encoding
