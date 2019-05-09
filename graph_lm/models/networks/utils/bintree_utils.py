import math

import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np
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

def flat_infix_indices(depth):
    idx = infix_indices(depth=depth)
    lengths = np.power(2, np.arange(depth+1,dtype=np.int32))
    #print("lens: {}".format(lengths))
    offsets = np.cumsum(lengths)-lengths
    #print("offsets: {}".format(offsets))
    infix_idx = np.zeros(shape=(len(idx)), dtype=np.int32)
    for i, id in enumerate(idx):
        d, j = tree_index_map(id)
        infix_idx[i] = offsets[d]+j
    return infix_idx

def tree_index_map(idx):
    depth = len(idx)
    output_idx = 0
    mult = 1
    for i in reversed(idx):
        output_idx += mult * i
        mult *= 2
    return depth, output_idx


def stack_tree(outputs, indices):
    # outputs:[ (N,V), (N,2,V), (N,2,2,V)...]
    # indices:[ paths]

    slices = []
    for idx in indices:
        depth, output_idx = tree_index_map(idx)
        slice = outputs[depth][:, output_idx, :]
        slices.append(slice)
    stacked = tf.stack(slices, axis=0)  # (L,N,V)
    return stacked

def stack_tree_v2(outputs):
    idx = flat_infix_indices(len(outputs)-1)
    h = tf.concat(outputs, axis=1) # (N,L,D)
    h = tf.transpose(h, (1,0,2))
    h = tf.gather(params=h, indices=idx, axis=0)
    return h



def calc_children_resnet(x, hidden_dim, inputs=None, weights_regularizer=None, reuse=None):
    # X: (N,x, D)
    # Y: (N,2x,D)
    input_dim = x.shape[-1].value
    n = tf.shape(x)[0]
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
        kids = tf.stack([h_left, h_right], axis=2) #(N, L, 2, D)
        kids = tf.reshape(kids, (n, kids.shape[1].value * 2, input_dim))
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
        activation_fn=tf.nn.leaky_relu,
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


def binary_tree_upward_messages(inp, hidden_dim):
    """

    :param inp: (n, 2L, D)
    :param hidden_dim:
    :return: (n, L, D)
    """
    n = tf.shape(inp)[0]
    l2 = inp.shape[1].value
    l = l2 // 2
    d = inp.shape[2].value

    k = 2
    inp_rs = tf.reshape(inp, (n, l, k, d))
    h = tf.concat([inp_rs, branch_indicators(n, l, k)], axis=-1)
    h = slim.fully_connected(
        inputs=h,
        num_outputs=hidden_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='up_mlp_1'
    )  # (n,l, 2, d)
    h = slim.fully_connected(
        inputs=h,
        num_outputs=hidden_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='up_mlp_2'
    )  # (n,l, 2, d)
    h = tf.reduce_sum(h, axis=2)  # (n, l, d)
    return h


def binary_tree_upward(inputs, hidden_dim):
    with tf.variable_scope('binary_tree_upward'):
        msg_0 = tf.get_variable(
            shape=(1, 1, hidden_dim,),
            dtype=tf.float32,
            initializer=tf.initializers.zeros,
            name='msg_0'
        )
        msg_t = tf.tile(msg_0, (tf.shape(inputs[-1])[0], inputs[-1].shape[1].value, 1))
        msgs = [msg_t]
        for t, input_t in enumerate(reversed(inputs[1:])):
            with tf.variable_scope('binary_tree_upward_pass', reuse=t > 0):
                inp = tf.concat([input_t, msg_t], axis=-1)
                msg_t = binary_tree_upward_messages(
                    inp=inp,
                    hidden_dim=hidden_dim
                )
                msgs.append(msg_t)
        msgs.reverse()
        assert len(inputs) == len(msgs)
        for i, m in zip(inputs, msgs):
            assert i.shape[1].value == m.shape[1].value
        return msgs


def branch_indicators(n , l, k):
    ind = tf.eye(k, dtype=tf.float32)
    ind = tf.reshape(ind, (1, 1, k, k))
    ind = tf.tile(ind, [n, l, 1, 1])
    return ind

def binary_tree_downward_messages(inp, hidden_dim):
    """

    :param inp: (n, L, D)
    :param hidden_dim:
    :return: (n, 2L, D)
    """
    n = tf.shape(inp)[0]
    l = inp.shape[1].value
    l2 = l * 2
    d = inp.shape[2].value
    k = 2
    inp_rs = tf.reshape(inp, (n, l, 1, d))
    inp_rs = tf.tile(inp_rs, (1, 1, k, 1))
    h = tf.concat([inp_rs, branch_indicators(n,l,k)], axis=-1)  # (n, l, k, d)
    h = slim.fully_connected(
        inputs=h,
        num_outputs=hidden_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='down_mlp_1'
    )  # (n,l, 2, d)
    h = slim.fully_connected(
        inputs=h,
        num_outputs=hidden_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='down_mlp_2'
    )  # (n,l, 2, d)
    h = tf.reshape(h, (n, l2, h.shape[-1].value))
    return h


def binary_tree_downward(inputs, hidden_dim):
    with tf.variable_scope('binary_tree_downward'):
        msg_0 = tf.get_variable(
            shape=(1, 1, hidden_dim,),
            dtype=tf.float32,
            initializer=tf.initializers.zeros,
            name='msg_0'
        )
        msg_t = tf.tile(msg_0, (tf.shape(inputs[0])[0], inputs[0].shape[1].value, 1))
        msgs = [msg_t]
        for t, input_t in enumerate(inputs[:-1]):
            with tf.variable_scope('binary_tree_downward_pass', reuse=t > 0):
                inp = tf.concat([input_t, msg_t], axis=-1)
                msg_t = binary_tree_downward_messages(
                    inp=inp,
                    hidden_dim=hidden_dim
                )
                msgs.append(msg_t)
        assert len(inputs) == len(msgs)
        for i, m in zip(inputs, msgs):
            assert i.shape[1].value == m.shape[1].value
        return msgs


def concat_layers(*layers, axis=-1):
    return [
        tf.concat(ls, axis=axis)
        for ls in zip(*layers)
    ]


if __name__ == '__main__':
    idx = infix_indices(depth=2)
    tree_idx = [tree_index_map(i) for i in idx]
    print(idx)
    print(tree_idx)
