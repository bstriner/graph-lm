import tensorflow as tf
from tensorflow.contrib import slim

from ..rnn_util import lstm
from ...stats import get_bias


def make_dag(latent, sequence_lengths, params):
    """

    :param latent: (N, L, D)
    :param sequence_lengths: (N,)
    :return:
    """
    n = tf.shape(latent)[0]
    L = tf.shape(latent)[1]
    with tf.variable_scope('dag'):
        proj_a = slim.fully_connected(
            inputs=latent,
            activation_fn=tf.nn.leaky_relu,
            num_outputs=params.attention_dim,
            scope='proj_a'
        )
        proj_b = slim.fully_connected(
            inputs=latent,
            activation_fn=tf.nn.leaky_relu,
            num_outputs=params.attention_dim,
            scope='proj_b'
        )

    proj_a_ta = tf.TensorArray(size=n, dtype=proj_a.dtype).unstack(proj_a)
    proj_b_ta = tf.TensorArray(size=n, dtype=proj_b.dtype).unstack(proj_b)
    sequence_lengths_ta = tf.TensorArray(size=n, dtype=sequence_lengths.dtype).unstack(sequence_lengths)

    def body(t, output_ta_t: tf.TensorArray, penalty_ta_t: tf.TensorArray):
        proj_a_t = proj_a_ta.read(t)
        proj_b_t = proj_b_ta.read(t)
        sequence_lengths_t = sequence_lengths_ta.read(t)
        proj_a_t_slice = proj_a_t[:sequence_lengths_t, :]
        proj_b_t_slice = proj_b_t[:sequence_lengths_t, :]
        energy = tf.tensordot(proj_a_t_slice, proj_b_t_slice, axes=[(1,), (1,)])
        # energy = energy*(1-tf.eye(sequence_lengths_t)) # mask diagonal
        # edges = tf.nn.sigmoid(energy)
        # edges = edges * (1 - tf.eye(sequence_lengths_t))  # mask diagonal
        edges = tf.nn.softmax(energy, axis=-1)
        # exp_edges = edges
        # for _ in range(params.series_depth):
        #    exp_edges = tf.matmul(exp_edges, edges)
        exp_edges = tf.linalg.expm(input=edges)
        # penalty_t = tf.maximum(tf.trace(exp_edges) - tf.cast(sequence_lengths_t, tf.float32), 0)
        penalty_t = tf.trace(exp_edges) - tf.cast(sequence_lengths_t, tf.float32)
        # penalty_t = tf.reduce_sum(tf.maximum(tf.trace(exp_edges) - tf.cast(sequence_lengths_t, tf.float32), 0))

        length_diff = L - sequence_lengths_t
        edges_padded = tf.pad(
            tensor=edges,
            paddings=[(0, length_diff), (0, length_diff)]
        )
        output_ta_t1 = output_ta_t.write(value=edges_padded, index=t)
        penalty_ta_t1 = penalty_ta_t.write(value=penalty_t, index=t)
        return t + 1, output_ta_t1, penalty_ta_t1

    def cond(t, output_ta_t: tf.TensorArray, penalty_ta_t: tf.TensorArray):
        return t < n

    output_ta = tf.TensorArray(size=n, dtype=tf.float32)
    penalty_ta = tf.TensorArray(size=n, dtype=tf.float32)
    _, output_ta, penalty_ta = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(tf.constant(0, dtype=tf.int32), output_ta, penalty_ta)
    )
    dag = output_ta.stack()  # (N, L, L)
    penalty = penalty_ta.stack()  # (N,)
    penalty = tf.check_numerics(penalty, message='penalty numerics')
    return dag, penalty


def make_message(inputs, params):
    h = inputs
    h = slim.fully_connected(
        inputs=h,
        activation_fn=tf.nn.leaky_relu,
        num_outputs=params.decoder_dim,
        scope='messages_1'
    )
    messages = slim.fully_connected(
        inputs=h,
        activation_fn=tf.nn.leaky_relu,
        num_outputs=params.decoder_dim,
        scope='messages_2'
    )
    return messages


def pass_messge(messages, dag_bw):
    """

    :param messages: (N, L, D)
    :param dag_bw: (N, L, L)
    :return:
    """
    return tf.matmul(
        dag_bw, messages
    )


def message_passing(
        latent, dag_bw, params
):
    """

    :param latent: (N, L, Dlatent)
    :param dag_bw: (N, L, L)
    :param sequence_lengths: (N,)
    :return:
    """
    # print("dag_bw: {}".format(dag_bw))
    n = tf.shape(latent)[0]
    l = tf.shape(latent)[1]
    messages_t = tf.zeros(
        dtype=tf.float32,
        shape=(n, l, params.decoder_dim)
    )
    inputs_t = tf.concat([latent, messages_t], axis=-1)
    for t in range(params.message_depth):
        with tf.variable_scope('messages', reuse=t > 0):
            # print("inputs_{}: {}".format(t, inputs_t))
            messages_t = make_message(inputs_t, params=params)
            # print("messages1_{}: {}".format(t, messages_t))
            messages_t = pass_messge(messages_t, dag_bw=dag_bw)
            # print("messages2_{}: {}".format(t, messages_t))
            inputs_t = tf.concat([latent, messages_t], axis=-1)
    # inputs_t (N, L, Dlatent+decoder_dim)
    return inputs_t


def process_latent(latent, sequence_lengths, params):
    latent, _ = lstm(
        x=tf.transpose(latent, (1, 0, 2)),
        num_layers=2,
        num_units=params.decoder_dim,
        bidirectional=True,
        sequence_lengths=sequence_lengths
    )
    latent = tf.transpose(latent, (1, 0, 2))
    return latent


def vae_decoder_dag(latent, sequence_lengths, vocab_size, params, n, weights_regularizer=None, is_training=True):
    # latent (N, L, D)
    with tf.variable_scope('decoder'):
        latent_processed = process_latent(latent=latent, sequence_lengths=sequence_lengths, params=params)

        dag, penalty = make_dag(
            latent=latent_processed,
            sequence_lengths=sequence_lengths,
            params=params
        )

        # dag_bw = tf.transpose(dag, (0, 2, 1))
        hidden = message_passing(
            latent=latent,
            dag_bw=dag,
            params=params
        )
        # hidden (N, L, Dlatent+decoder_dim)

        h = slim.fully_connected(
            inputs=hidden,
            activation_fn=tf.nn.leaky_relu,
            num_outputs=params.decoder_dim,
            scope='output_1'
        )
        logits = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size,
            activation_fn=None,
            scope='output_2',
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.initializers.constant(
                value=get_bias(smoothing=params.bias_smoothing),
                verify_shape=True)
        )  # (N,L,V)
        return logits, penalty
