import math

import tensorflow as tf
from tensorflow.contrib import slim


def binary_tree_upward_messages_v2(inp, hidden_dim):
    """

    :param inp: (n, 2L, D)
    :param hidden_dim:
    :return: (n, L, D)
    """
    n = tf.shape(inp)[0]
    lk = inp.shape[1].value
    k = 2
    l = lk // k
    h = inp
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
    h = tf.reshape(
        h, (n, l, k*hidden_dim)
    )
    return h


def binary_tree_upward_v2(inputs, hidden_dim):
    with tf.variable_scope('binary_tree_upward'):
        msg_0 = tf.get_variable(
            shape=(1, 1, 1, hidden_dim,),
            dtype=tf.float32,
            initializer=tf.initializers.zeros,
            name='msg_0'
        )
        k = 2
        n = tf.shape(inputs[-1])[0]
        msg_t = tf.tile(msg_0, (n, inputs[-1].shape[1].value, k, 1))
        msg_t = tf.reshape(msg_t, (n, inputs[-1].shape[1].value, hidden_dim*k))
        msgs = [msg_t]
        for t, input_t in enumerate(reversed(inputs[1:])):
            with tf.variable_scope('binary_tree_upward_pass', reuse=t > 0):
                inp = tf.concat([input_t, msg_t], axis=-1)
                msg_t = binary_tree_upward_messages_v2(
                    inp=inp,
                    hidden_dim=hidden_dim
                )
                msgs.append(msg_t)
        msgs.reverse()
        assert len(inputs) == len(msgs)
        for i, m in zip(inputs, msgs):
            assert i.shape[1].value == m.shape[1].value
        return msgs


def binary_tree_downward_messages_v2(inp, hidden_dim):
    """

    :param inp: (n, L, D)
    :param hidden_dim:
    :return: (n, 2L, D)
    """
    n = tf.shape(inp)[0]
    l = inp.shape[1].value
    k = 2
    lk = l * k
    h = inp
    h = slim.fully_connected(
        inputs=h,
        num_outputs=hidden_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='down_mlp_1'
    )  # (n,l, d)
    h = slim.fully_connected(
        inputs=h,
        num_outputs=hidden_dim*k,
        activation_fn=tf.nn.leaky_relu,
        scope='down_mlp_2'
    )  # (n,l, d*k)
    h = tf.reshape(h, (n, lk, hidden_dim)) # (n, l*k, d)
    return h


def binary_tree_downward_v2(inputs, hidden_dim):
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
                msg_t = binary_tree_downward_messages_v2(
                    inp=inp,
                    hidden_dim=hidden_dim
                )
                msgs.append(msg_t)
        assert len(inputs) == len(msgs)
        for i, m in zip(inputs, msgs):
            assert i.shape[1].value == m.shape[1].value
        return msgs

