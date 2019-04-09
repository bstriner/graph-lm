import tensorflow as tf

from graph_lm.models.networks.utils.dag_utils import message_passing
from ...sn import sn_fully_connected, sn_kernel

def discriminator_dag_supervised(
        latent, dag, dag_bw, params, idx,
        tags, tag_size, weights_regularizer=None,
        is_training=True):
    # latent (N, L, D)
    with tf.variable_scope('decoder'):
        with tf.variable_scope('tag_embedding'):
            tag_embeddings = sn_kernel(
                shape=[tag_size, params.decoder_dim],
                scope="tag_embeddings"
            )
        h_tags = tf.nn.embedding_lookup(params=tag_embeddings, ids=tags)
        h = tf.concat([latent, h_tags], axis=-1)
        with tf.variable_scope("forward"):
            h = message_passing(
                latent=h,
                dag_bw=dag_bw,
                params=params,
                fully_connected_fn=sn_fully_connected
            )
        with tf.variable_scope("backward"):
            h = message_passing(
                latent=h,
                dag_bw=dag,
                params=params,
                fully_connected_fn=sn_fully_connected
            )
        with tf.variable_scope('output_mlp'):
            h = sn_fully_connected(
                inputs=h,
                activation_fn=tf.nn.leaky_relu,
                num_outputs=params.decoder_dim,
                scope='output_1',
                weights_regularizer=weights_regularizer
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
            )  # (n,)
        return dis_values  # (n,)
