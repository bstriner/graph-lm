import math

import tensorflow as tf
from tensorflow.contrib import slim


def infix_indices(depth, stack=[]):
    if depth >= 0:
        left = infix_indices(depth - 1, stack + [0])
        right = infix_indices(depth - 1, stack + [1])
        middle = stack
        indices = left + [middle] + right
        assert len(indices) == int(math.pow(2, depth + 1) - 1)
        return indices
    else:
        return []


def stack_tree(outputs, indices):
    # outputs:[ (N,V), (N,2,V), (N,2,2,V)...]
    # indices:[ paths]

    slices = []
    for idx in indices:
        depth = len(idx)
        output = outputs[depth]
        output_idx = 0
        mult = 1
        for i in reversed(idx):
            output_idx += mult * i
            mult *= 2
        slices.append(output[:, output_idx, :])
    stacked = tf.stack(slices, axis=0)  # (L,N,V)
    return stacked


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


def calc_children(x, hidden_dim, inputs=None, weights_regularizer=None, reuse=None):
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('children_mlp', reuse=reuse):
        assert x.shape.ndims == 3
        n = tf.shape(x)[0]
        input_len = x.shape[1].value
        output_dim = x.shape[2].value
        assert input_len
        assert output_dim
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
        h = slim.fully_connected(
            inputs=h,
            num_outputs=2 * output_dim,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer
        )
        kids = tf.reshape(h, (n, 2 * input_len, output_dim))  # params.decoder_dim))
        print("Calc child: {}->{}".format(x, kids))
        return kids


def calc_parents(x, hidden_dim, inputs=None, weights_regularizer=None, reuse=None):
    """

    :param x: (N,2X,D)
    :param hidden_dim:
    :param inputs: (N,X,D)
    :param weights_regularizer:
    :param reuse:
    :return: (N,X,D)
    """
    print("calc_parents: {},{},{}".format(x, inputs, hidden_dim))
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('parent_mlp', reuse=reuse):
        assert x.shape.ndims == 3
        n = tf.shape(x)[0]
        input_len = x.shape[1].value
        input_dim = x.shape[2].value
        output_len = input_len // 2
        assert input_len
        assert input_len % 2 == 0
        assert input_dim
        x_parent = tf.reshape(x, (n, output_len, input_dim * 2))
        if inputs is not None:
            h = tf.concat([x_parent, inputs], axis=-1)
        else:
            h = x_parent
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
        h = slim.fully_connected(
            inputs=h,
            num_outputs=input_dim,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer
        )
        parents = h
        print("Calc parents: {}->{}".format(x, parents))
        return parents


def binary_tree_resnet(x0, depth, hidden_dim, inputs=None):
    if inputs is not None:
        assert len(inputs) == depth + 1
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


def binary_tree_down(x0, depth, hidden_dim, inputs=None):
    if inputs is not None:
        assert inputs[0].shape[1].value == 1
        assert len(inputs) == depth + 1
    h = tf.expand_dims(x0, 1)
    if inputs is not None:
        h = tf.concat([h, inputs[0]], axis=-1)
    h = slim.fully_connected(
            inputs=h,
            num_outputs=hidden_dim,
            activation_fn=None,
            scope='down_x0'
    )
    layers = [h]
    for i in range(depth):
        h = calc_children(
            x=h,
            hidden_dim=hidden_dim,
            inputs=None if inputs is None else inputs[i],
            reuse=i > 0
        )
        layers.append(h)
    return layers


def binary_tree_up(inputs, hidden_dim):
    print("binary_tree_up: {}".format(inputs))
    assert inputs[0].shape[1].value == 1
    xt = slim.fully_connected(
            inputs=inputs[-1],
            num_outputs=hidden_dim,
            activation_fn=None,
            scope='up_xt'
    )
    outputs = [xt]
    for t, input_t in enumerate(reversed(inputs[:-1])):
        xt = calc_parents(
            x=xt,
            inputs=input_t,
            reuse=t > 0,
            hidden_dim=hidden_dim
        )
        outputs.append(xt)
    outputs.reverse()
    return outputs
