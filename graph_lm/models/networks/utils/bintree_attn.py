import tensorflow as tf
from tensorflow.contrib import slim

from ..embed_sentences import embed_sentences
from .attn_util import calc_attn_v2
from .bintree_utils import binary_tree_downward, binary_tree_upward, branch_indicators, concat_layers, \
    infix_indices, stack_tree
from .rnn_util import lstm


def calc_layer_inputs(h, params, input_hidden_t, input_proj, sequence_lengths):
    query = slim.fully_connected(
        inputs=h,
        num_outputs=params.attention_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='query_projection'
    )
    attn = calc_attn_v2(
        a=query, b=input_proj, b_lengths=sequence_lengths,
        a_transpose=False,
        b_transpose=False
    )
    ctx = tf.matmul(
        attn,  # (n, ol, il)
        input_hidden_t)  # (n, il, d)
    inp = tf.concat([h, ctx], axis=-1)  # (n, il, d)
    return inp, attn


def calc_layer_outputs(inp, params):
    output = slim.fully_connected(
        inputs=inp,
        num_outputs=params.encoder_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='encoder_attn_output_1'
    )
    output = slim.fully_connected(
        inputs=output,
        num_outputs=params.encoder_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='encoder_attn_output_2'
    )
    return output

def calc_layer_children(
        inp,
        params):
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('children_mlp'):
        assert inp.shape.ndims == 3
        n = tf.shape(inp)[0]
        input_len = inp.shape[1].value
        input_dim = inp.shape[2].value
        k = 2
        assert input_len
        assert input_dim
        h = inp
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim*k,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2'
        )
        h = tf.reshape(h, (n, input_len * k, params.encoder_dim))
        return h


def binary_tree_down_recurrent_attention(
        x0,
        input_hidden,
        sequence_lengths,
        tree_depth,
        params):
    """

    :param x0:
    :param input_hidden:
    :param sequence_lengths:
    :param tree_depth:
    :param params:
    :return:
    """
    input_hidden_t = tf.transpose(input_hidden, (1, 0, 2))  # (n, o, d)
    input_proj = slim.fully_connected(
        inputs=input_hidden_t,
        num_outputs=params.attention_dim,
        activation_fn=tf.nn.leaky_relu,
        scope='encoder_input_proj'
    )  # (N,O, D)
    h = tf.expand_dims(x0, 1)  # (N,L,D)
    layers = []
    attns = []
    with tf.variable_scope("bintree", reuse=False) as treescope:
        inp, attn = calc_layer_inputs(
            h=h,
            params=params,
            input_hidden_t=input_hidden_t,
            input_proj=input_proj,
            sequence_lengths=sequence_lengths)
        #output = calc_layer_outputs(
        #    inp=inp,
        #    params=params)
        layers.append(inp)
        attns.append(attn)
        h = calc_layer_children(
            inp=inp,
            params=params
        )
    for i in range(tree_depth):
        with tf.variable_scope(treescope, reuse=True):
            inp, attn = calc_layer_inputs(
                h=h,
                params=params,
                input_hidden_t=input_hidden_t,
                input_proj=input_proj,
                sequence_lengths=sequence_lengths)
            #output = calc_layer_outputs(
            #    inp=inp,
            #    params=params)
            layers.append(inp)
            attns.append(attn)
            if i < tree_depth - 1:
                h = calc_layer_children(
                    inp=inp,
                    params=params
                )

    assert len(layers) == tree_depth + 1
    assert len(attns) == tree_depth + 1
    attn_idx = infix_indices(tree_depth)
    flat_attns = stack_tree(attns, indices=attn_idx)  # (L,N,V)
    attn_img = tf.expand_dims(tf.transpose(flat_attns, (1, 0, 2)), axis=3)
    tf.summary.image('encoder_attention', attn_img)
    return layers

