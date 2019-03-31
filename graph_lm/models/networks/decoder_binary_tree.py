import tensorflow as tf
from tensorflow.contrib import slim

from .tree_utils import infix_indices, stack_tree


def calc_children(x, params, weights_regularizer=None):
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('children_mlp', reuse=tf.AUTO_REUSE):
        h = x
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=2 * params.decoder_dim,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer
        )
        kids = tf.reshape(h, (tf.shape(h)[0], 2 * h.shape[1].value, params.decoder_dim))  # params.decoder_dim))
        print("Calc child: {}->{}".format(x, kids))
        return kids


def calc_output(x, vocab_size, params, weights_regularizer=None):
    # X: (N,*, D)
    # Y: (N,*, V)
    with tf.variable_scope('output_mlp', reuse=tf.AUTO_REUSE):
        h = x
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer
        )
    return h


def decoder_binary_tree(latent, vocab_size, params, weights_regularizer=None):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0
    h = slim.fully_connected(
        latent,
        num_outputs=params.decoder_dim,
        scope='projection',
        activation_fn=None,
        weights_regularizer=weights_regularizer
    )
    h = tf.expand_dims(h, axis=1)

    tree_layers = []
    tree_layers.append(h)
    for i in range(depth):
        h = calc_children(h, params=params, weights_regularizer=weights_regularizer)
        tree_layers.append(h)

    indices = infix_indices(depth)
    flat_layers = stack_tree(tree_layers, indices=indices)  # (L,N,V)
    logits = calc_output(
        flat_layers,
        vocab_size=vocab_size,
        params=params,
        weights_regularizer=weights_regularizer)
    return tree_layers, logits
