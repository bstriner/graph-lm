import tensorflow as tf
from tensorflow.contrib import slim


def calc_children_resnet(x, hidden_dim, inputs=None, weights_regularizer=None, reuse=None):
    # X: (N,x, D)
    # Y: (N,2x,D)
    input_dim = x.shape[-1].value
    with tf.variable_scope('children_mlp', reuse=reuse):
        if inputs is not None:
            h = tf.concat([x, inputs], axis=-1)
        else:
            h = x
        h = slim.fully_connected(
            inputs=h,
            num_outputs=hidden_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=hidden_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2',
            weights_regularizer=weights_regularizer
        )
        h_left = x + slim.fully_connected(
            inputs=h,
            num_outputs=input_dim,
            activation_fn=None,
            scope='output_mlp_left',
            weights_regularizer=weights_regularizer
        )  # (N,X,L)
        h_right = x + slim.fully_connected(
            inputs=h,
            num_outputs=input_dim,
            activation_fn=None,
            scope='output_mlp_right',
            weights_regularizer=weights_regularizer
        )  # (N,X,L)

        # h_base = tf.tile(x, (1, 1, 2))
        # kids = tf.reshape(h + h_base,
        #                  (tf.shape(h)[0], 2 * h.shape[1].value, params.decoder_dim))  # params.decoder_dim))
        kids = tf.stack([h_left, h_right], axis=2)
        kids = tf.reshape(kids, (-1, kids.shape[1].value * 2, input_dim))
        print("Calc child: {}->{}".format(x, kids))
        return kids


def calc_children(x, params, weights_regularizer=None, reuse=None):
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('children_mlp', reuse=reuse):
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


def binary_tree_resnet(x0, depth, hidden_dim, inputs=None):
    h = tf.expand_dims(x0, 1)
    layers = [h]
    for i in range(depth):
        h = calc_children_resnet(
            x=h,
            hidden_dim=hidden_dim,
            inputs=None if inputs is None else inputs[i],
            reuse=i > 0
        )
        layers.append(h)
    return layers
