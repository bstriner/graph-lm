import tensorflow as tf
from tensorflow.contrib import slim

from .embed_sentences import embed_sentences
from .utils.attn_util import calc_attn_v2
from .utils.bintree_utils import binary_tree_downward, binary_tree_upward, branch_indicators, concat_layers, \
    infix_indices, stack_tree
from .utils.rnn_util import lstm


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


def calc_layer_children_old(
        inp,
        params):
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('children_mlp'):
        assert inp.shape.ndims == 3
        n = tf.shape(inp)[0]
        input_len = inp.shape[1].value
        output_dim = inp.shape[2].value
        assert input_len
        assert output_dim
        h = inp
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim * 2,
            activation_fn=None,
            scope='output_mlp_3'
        )
        kids = tf.reshape(h, (n, 2 * input_len, params.decoder_dim))  # ))
        return kids


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
        h = tf.reshape(h, (n, input_len, 1, input_dim))
        h = tf.tile(h, (1, 1, k, 1))
        h = tf.concat([h, branch_indicators(n, input_len, k)], axis=-1)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_3'
        )
        h = tf.reshape(h, (n, input_len * 2, h.shape[-1].value))
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
    # h = slim.fully_connected(
    #    inputs=h,
    #    num_outputs=params.encoder_dim,
    #    activation_fn=tf.nn.leaky_relu,
    #    scope='encoder_input_h0'
    # )  # (N,O, D)
    layers = []
    attns = []
    with tf.variable_scope("bintree", reuse=False) as treescope:
        inp, attn = calc_layer_inputs(
            h=h,
            params=params,
            input_hidden_t=input_hidden_t,
            input_proj=input_proj,
            sequence_lengths=sequence_lengths)
        output = calc_layer_outputs(
            inp=inp,
            params=params)
        layers.append(output)
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
            output = calc_layer_outputs(
                inp=inp,
                params=params)
            layers.append(output)
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


def encoder_bintree_recurrent_attn_base(
        inputs, token_lengths, params,
        weights_regularizer=None, is_training=True):
    """

    :param inputs: (L,N)
    :param token_lengths:
    :return:
    """
    n = tf.shape(inputs)[1]
    with tf.variable_scope('input_lstm'):
        h = inputs
        hidden_state, hidden_state_final = lstm(
            x=h,
            num_layers=3,
            num_units=params.encoder_dim,
            bidirectional=True,
            sequence_lengths=token_lengths
        )
        h = tf.concat(hidden_state_final, axis=-1)  # (layers*directions, N, D)
        h = tf.transpose(h, (1, 0, 2))  # (N,layers*directions,D)
        h = tf.reshape(h, (n, h.shape[1].value * h.shape[2].value))  # (N, layers*directions*D)
        if params.batch_norm:
            h = slim.batch_norm(inputs=h, is_training=is_training)
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_1',
            weights_regularizer=weights_regularizer
        )
        if params.batch_norm:
            h = slim.batch_norm(inputs=h, is_training=is_training)
        flat_encoding = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_3',
            weights_regularizer=weights_regularizer
        )  # (N,D)
    with tf.variable_scope('bintree_attention'):
        hs = binary_tree_down_recurrent_attention(
            x0=flat_encoding,
            input_hidden=hidden_state,
            sequence_lengths=token_lengths,
            tree_depth=params.tree_depth,
            params=params)
    with tf.variable_scope('encoder_bintree_up'):
        messages_up = binary_tree_upward(
            hidden_dim=params.encoder_dim,
            inputs=hs
        )
    with tf.variable_scope('encoder_bintree_down'):
        hs = concat_layers(hs, messages_up)
        messages_down = binary_tree_downward(
            hidden_dim=params.encoder_dim,
            inputs=hs
        )
        hs = concat_layers(hs, messages_down)
    return hs


def encoder_bintree_recurrent_attn_vae(
        tokens, token_lengths, vocab_size, params, n, weights_regularizer=None, is_training=True):
    """

    :param tokens: (N,L)
    :param token_lengths: (N,)
    :param vocab_size:
    :param params:
    :param n:
    :param output_length:
    :param weights_regularizer:
    :return:
    """
    with tf.variable_scope('encoder'):
        with tf.variable_scope('embedding'):
            h = tf.transpose(tokens, (1, 0))  # (L,N)
            h = embed_sentences(
                tokens=h,
                token_lengths=token_lengths,
                vocab_size=vocab_size,
                params=params
            )
        with tf.variable_scope('base'):
            hs = encoder_bintree_recurrent_attn_base(
                inputs=h,
                token_lengths=token_lengths,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training
            )
        with tf.variable_scope('output_projection'):
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h1',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h2',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            mu = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.latent_dim,
                    activation_fn=None,
                    scope='encoder_mlp_mu',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            logsigma = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.latent_dim,
                    activation_fn=None,
                    scope='encoder_mlp_logsigma',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            return mu, logsigma


def encoder_bintree_recurrent_attn_aae(
        tokens, token_lengths, noise, vocab_size, params, n, weights_regularizer=None, is_training=True):
    """

    :param tokens: (N,L)
    :param token_lengths: (N,)
    :param vocab_size:
    :param params:
    :param n:
    :param output_length:
    :param weights_regularizer:
    :return:
    """
    with tf.variable_scope('encoder'):
        with tf.variable_scope('embedding'):
            h = tf.transpose(tokens, (1, 0))  # (L,N)
            h = embed_sentences(
                tokens=h,
                token_lengths=token_lengths,
                vocab_size=vocab_size,
                params=params
            )
            noise_t = tf.transpose(noise, (1, 0, 2))
            h = tf.concat([h, noise_t], axis=-1)
        with tf.variable_scope('base'):
            hs = encoder_bintree_recurrent_attn_base(
                inputs=h,
                token_lengths=token_lengths,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training
            )
        with tf.variable_scope('output_projection'):
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h1',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            hs = [
                slim.fully_connected(
                    inputs=enc,
                    num_outputs=params.encoder_dim,
                    activation_fn=tf.nn.leaky_relu,
                    scope='encoder_mlp_output_projection_h2',
                    weights_regularizer=weights_regularizer,
                    reuse=i > 0
                )
                for i, enc in enumerate(hs)
            ]
            return hs
