import math

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM, CUDNN_RNN_BIDIRECTION

from ..kl import kl
from ..sparse import sparsify


def vae_encoder(tokens, token_lengths, vocab_size, params, n):
    with tf.variable_scope('encoder'):
        h = tf.transpose(tokens, (1, 0))  # (L,N)
        embeddings = tf.get_variable(
            dtype=tf.float32,
            name="embeddings",
            shape=[vocab_size, params.encoder_dim],
            initializer=tf.initializers.truncated_normal(
                stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
        h = tf.nn.embedding_lookup(embeddings, h)  # (L, N, D)
        encoder_lstm = CudnnLSTM(num_layers=3, num_units=params.encoder_dim, direction=CUDNN_RNN_BIDIRECTION)
        _, h = encoder_lstm(h, sequence_lengths=token_lengths)
        print("h1: {}".format(h))
        h = h[0][-2:, :, :]  # (2, N, D)
        print("h2: {}".format(h))
        h = tf.transpose(h, (1, 0, 2))  # (N,2,D)
        print("h3: {}".format(h))
        h = tf.reshape(h, (n, 2 * h.shape[-1].value))  # (N, 2D)
        print("h4: {}".format(h))
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_1'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='encoder_mlp_2'
        )
        mu = slim.fully_connected(
            inputs=h,
            num_outputs=params.latent_dim,
            activation_fn=None,
            scope='encoder_mlp_mu'
        )
        logsigma = slim.fully_connected(
            inputs=h,
            num_outputs=params.latent_dim,
            activation_fn=None,
            scope='encoder_mlp_logsigma'
        )
        return mu, logsigma


def calc_output(x, vocab_size, params):
    # X: (N,*, D)
    # Y: (N,*, V)
    with tf.variable_scope('output_mlp', reuse=tf.AUTO_REUSE):
        h = x
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='output_mlp_3'
        )
    return h


def calc_children(x, params):
    # X: (N,x, D)
    # Y: (N,2x,D)
    with tf.variable_scope('children_mlp', reuse=tf.AUTO_REUSE):
        h = x
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_1'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.decoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='output_mlp_2'
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=2 * params.decoder_dim,
            activation_fn=None,
            scope='output_mlp_3'
        )
        kids = tf.reshape(h, (tf.shape(h)[0], 2 * h.shape[1].value, params.decoder_dim))  # params.decoder_dim))
        print("Calc child: {}->{}".format(x, kids))
        return kids


def infix_indices(depth, stack=[]):
    if depth >= 0:
        left = infix_indices(depth - 1, stack + [0])
        right = infix_indices(depth - 1, stack + [1])
        middle = stack
        return left + [middle] + right
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


def vae_decoder(latent, vocab_size, params):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0
    h = slim.fully_connected(
        latent,
        num_outputs=params.decoder_dim,
        scope='projection',
        activation_fn=None
    )
    h = tf.expand_dims(h, axis=1)

    layers = []
    layers.append(h)
    for i in range(depth):
        h = calc_children(h, params=params)
        layers.append(h)
    return layers


def make_model_vae_binary_tree(
        run_config,
        vocab
):
    vocab_size = vocab.shape[0]
    print("Vocab size: {}".format(vocab_size))

    def model_fn(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        # Inputs
        tokens = features['features']  # (N, L)
        token_lengths = features['feature_length']  # (N,)
        sequence_mask = tf.sequence_mask(maxlen=tf.shape(tokens)[1], lengths=token_lengths)
        n = tf.shape(tokens)[0]
        depth = params.tree_depth

        with tf.control_dependencies([
            tf.assert_greater_equal(tf.pow(2, depth + 1) - 1, token_lengths, message="Tokens longer than tree size"),
            tf.assert_less_equal(tokens, vocab_size - 1, message="Tokens larger than vocab"),
            tf.assert_greater_equal(tokens, 0, message="Tokens less than 0")
        ]):
            tokens = tf.identity(tokens)

        # Encoder
        mu, logsigma = vae_encoder(
            tokens=tokens,
            token_lengths=token_lengths,
            vocab_size=vocab_size,
            params=params,
            n=n
        )
        # Sampling
        latent_sample, latent_prior_sample = kl(
            mu=mu,
            logsigma=logsigma,
            params=params,
            n=n)

        # Decoder
        tree_layers = vae_decoder(
            latent=latent_sample,
            vocab_size=vocab_size,
            params=params)
        indices = infix_indices(depth)
        print("Lengths: {} vs {}".format(len(indices), math.pow(2, depth+1) - 1))
        assert len(indices) == int(math.pow(2, depth + 1) - 1)
        assert max(len(i) for i in indices) == depth
        assert len(tree_layers) == depth + 1
        flat_layers = stack_tree(tree_layers, indices=indices)  # (L,N,V)
        logits = calc_output(flat_layers, vocab_size=vocab_size, params=params)

        ctc_labels = sparsify(tokens, sequence_mask)
        # ctc_labels = tf.sparse_transpose(ctc_labels, (1,0))
        print("Labels: {}".format(ctc_labels))
        sequence_length_ctc = tf.tile(tf.shape(logits)[0:1], (n,))
        # tf.tile(tf.pow([2], depth), (n,))
        print("CTC: {}, {}, {}".format(ctc_labels, logits, sequence_length_ctc))
        ctc_loss_raw = tf.nn.ctc_loss(
            labels=ctc_labels,
            sequence_length=sequence_length_ctc,
            inputs=logits,
            # sequence_length=tf.shape(logits)[0],
            ctc_merge_repeated=False,
            # preprocess_collapse_repeated=False,
            # ctc_merge_repeated=True,
            # ignore_longer_outputs_than_inputs=False,
            time_major=True
        )
        ctc_loss = tf.reduce_mean(ctc_loss_raw)
        tf.losses.add_loss(ctc_loss)

        total_loss = tf.losses.get_total_loss()

        # tf.argmax(logits, axis=-1)
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            inputs=logits,
            sequence_length=sequence_length_ctc
        )
        generated = tf.sparse_tensor_to_dense(decoded[0])
        generated_indices = decoded[0].indices
        tf.summary.scalar('ctc_loss', ctc_loss)
        tf.summary.scalar('total_loss', total_loss)

        # Train
        optimizer = tf.train.AdamOptimizer(params.lr)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        eval_metric_ops = {
            'ctc_loss_eval': tf.metrics.mean(ctc_loss_raw),
            'token_lengths_eval': tf.metrics.mean(token_lengths)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metric_ops,
            train_op=train_op)

    return model_fn
