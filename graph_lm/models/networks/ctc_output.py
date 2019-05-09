import math

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import common_shapes

from ...stats import get_bias_ctc


def fc_shared_weight(inputs, vocab_size, embeddings, biases_initializer, activation_fn=None):
    with tf.variable_scope('fc_shared_weights'):
        word_weight = tf.transpose(embeddings, (1, 0))
        embedding_dim = embeddings.shape[-1].value
        assert embedding_dim
        input_dim = inputs.shape[-1].value
        assert input_dim == embedding_dim
        blank_weight = tf.get_variable(
            name='blank_weight',
            initializer=tf.initializers.random_normal(0.05),
            dtype=tf.float32,
            shape=(embedding_dim, 1)
        )
        weight = tf.concat([blank_weight, word_weight], axis=-1)
        print("fc_shared_weight: {}, {}, {}".format(blank_weight, word_weight, weight))
        bias = tf.get_variable(
            name='bias',
            initializer=biases_initializer,
            shape=(vocab_size + 1,),
            dtype=tf.float32
        )
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, weight, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            # if not context.executing_eagerly():
            #    shape = inputs.get_shape().as_list()
            #    output_shape = shape[:-1] + [self.units]
            #    outputs.set_shape(output_shape)
        else:
            # Cast the inputs to self.dtype, which is the variable dtype. We do not
            # cast if `should_cast_variables` is True, as in that case the variable
            # will be automatically casted to inputs.dtype.
            outputs = tf.matmul(inputs, weight)
        outputs = tf.nn.bias_add(outputs, bias)
        if activation_fn is not None:
            return activation_fn(outputs)  # pylint: disable=not-callable
        else:
            return outputs


def ctc_projection(inputs, params, vocab_size, embeddings=None, weights_regularizer=None):
    biases_initializer = tf.initializers.constant(get_bias_ctc(
        average_output_length=math.pow(2, params.tree_depth + 1) - 1,
        smoothing=0.05
    ))
    if embeddings is None:
        h = slim.fully_connected(
            inputs=inputs,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer,
            biases_initializer=biases_initializer)
    else:
        h = fc_shared_weight(
            inputs=inputs,
            vocab_size=vocab_size,
            embeddings=embeddings,
            biases_initializer=biases_initializer
        )
    assert h.shape[-1].value == vocab_size+1
    return h


def calc_ctc_output(x, vocab_size, params, weights_regularizer=None, reuse=False, is_training=True, embeddings=None):
    # X: (N,*, D)
    # Y: (N,*, V)
    with tf.variable_scope('output_mlp', reuse=reuse):
        h = x
        if params.batch_norm:
            h = slim.batch_norm(h, is_training=is_training)
        for i in range(params.decoder_layers):
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.decoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='decoder_output_mlp_{}'.format(i),
                weights_regularizer=weights_regularizer
            )
            if params.batch_norm:
                h = slim.batch_norm(h, is_training=is_training)
        h = ctc_projection(
            inputs=h,
            params=params,
            vocab_size=vocab_size,
            embeddings=embeddings,
            weights_regularizer=weights_regularizer
        )
        return h


def calc_ctc_output_resnet(x, vocab_size, params, weights_regularizer=None, reuse=False, is_training=True,
                           embeddings=None):
    # X: (N,*, D)
    # Y: (N,*, V)
    with tf.variable_scope('output_mlp_resnet', reuse=reuse):
        h = x
        for i in range(params.decoder_layers):
            if params.batch_norm:
                h = slim.batch_norm(h, is_training=is_training)
            h1 = slim.fully_connected(
                inputs=h,
                num_outputs=params.decoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='decoder_output_mlp_1_{}'.format(i),
                weights_regularizer=weights_regularizer
            )
            h2 = slim.fully_connected(
                inputs=h1,
                num_outputs=params.decoder_dim,
                activation_fn=None,
                scope='decoder_output_mlp_2_{}'.format(i),
                weights_regularizer=weights_regularizer
            )
            h = h + h2
        if h.shape[-1].value != params.embedding_dim:
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.embedding_dim,
                activation_fn=None,
                scope='decoder_output_embedding_projection',
                weights_regularizer=weights_regularizer
            )
        if params.batch_norm:
            h = slim.batch_norm(h, is_training=is_training)
        h = ctc_projection(
            inputs=h,
            params=params,
            vocab_size=vocab_size,
            embeddings=embeddings,
            weights_regularizer=weights_regularizer
        )
        return h
