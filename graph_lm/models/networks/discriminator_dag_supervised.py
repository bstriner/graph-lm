import tensorflow as tf

from graph_lm.models.networks.utils.dag_utils import message_passing
from ...sn import sn_fully_connected, sn_kernel

def discriminator_dag_supervised(
        latent, dag, dag_bw, dag_feats, sequence_length, params, idx, weights_regularizer=None,
        is_training=True):
    # latent (N, L, D)
    with tf.variable_scope('discriminator'):
        h = tf.concat([latent, dag_feats], axis=-1)
        with tf.variable_scope("upward"):
            h = message_passing(
                latent=h,
                dag_bw=dag_bw,
                params=params,
                fully_connected_fn=sn_fully_connected,
                weights_regularizer=weights_regularizer,
                hidden_depth=params.discriminator_layers,
                dim=params.discriminator_dim
            )
        with tf.variable_scope("downward"):
            h = message_passing(
                latent=h,
                dag_bw=dag,
                params=params,
                fully_connected_fn=sn_fully_connected,
                weights_regularizer=weights_regularizer,
                hidden_depth=params.discriminator_layers,
                dim=params.discriminator_dim
            )
        with tf.variable_scope('output_mlp'):
            for i in range(params.discriminator_layers):
                h = sn_fully_connected(
                    inputs=h,
                    activation_fn=tf.nn.leaky_relu,
                    weights_regularizer=weights_regularizer,
                    num_outputs=params.discriminator_dim,
                    scope='discriminator_output_{}'.format(i)
                )
            logits = sn_fully_connected(
                inputs=h,
                num_outputs=1,
                activation_fn=None,
                scope='output_2',
                weights_regularizer=weights_regularizer
            )  # (N,L,1)
            logits = tf.squeeze(logits, axis=-1)  # (N, L)
            logits_values = tf.gather_nd(
                params=logits,
                indices=idx
            )  # (X,)
            sparse_logits = tf.SparseTensor(
                values=logits_values,
                indices=tf.cast(idx, tf.int64),
                dense_shape=tf.cast(tf.shape(logits), tf.int64)
            )
            sparse_logits = tf.sparse_reorder(sparse_logits)
            dis_values = tf.sparse_reduce_sum(
                sp_input=sparse_logits,
                axis=-1
            ) / tf.cast(sequence_length, tf.float32)  # (n,)
        return dis_values  # (n,)
