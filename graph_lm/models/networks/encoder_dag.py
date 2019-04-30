import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.networks.utils.dag_utils import message_passing


def encoder_dag(dag, dag_bw, dag_feats, text, vocab_size, params, weights_regularizer=None):
    with tf.variable_scope("encoder"):
        text_embeddings = tf.get_variable(
            dtype=tf.float32,
            name="text_embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        h_text = tf.nn.embedding_lookup(params=text_embeddings, ids=text)  # (N, L, D)
        h = tf.concat([h_text, dag_feats], axis=-1)
        with tf.variable_scope("upward"):
            h = message_passing(
                latent=h,
                dag_bw=dag_bw,
                params=params,
                dim=params.encoder_dim,
                hidden_depth=params.encoder_layers,
                weights_regularizer=weights_regularizer
            )
        with tf.variable_scope("downward"):
            h = message_passing(
                latent=h,
                dag_bw=dag,
                params=params,
                dim=params.encoder_dim,
                hidden_depth=params.encoder_layers,
                weights_regularizer=weights_regularizer
            )

        for i in range(params.encoder_layers):
            h = slim.fully_connected(
                inputs=h,
                activation_fn=tf.nn.leaky_relu,
                num_outputs=params.encoder_dim,
                scope='encoder_output_{}'.format(i),
                weights_regularizer=weights_regularizer
            )

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
