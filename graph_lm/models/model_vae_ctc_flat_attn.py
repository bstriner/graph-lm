import os

import tensorflow as tf
from tensorflow.contrib import slim

from .attn_util import calc_attn_v2
from .rnn_util import lstm
from ..callbacks.ctc_callback import CTCHook
from ..kl import kl
from ..sparse import sparsify
from ..stats import get_bias_ctc


def vae_flat_encoder_attn(tokens, token_lengths, vocab_size, params, n, output_length, weights_regularizer=None
                          , is_training=True):
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
    L = tf.shape(tokens)[1]
    with tf.variable_scope('encoder'):
        with tf.variable_scope('step_1'):
            h = tf.transpose(tokens, (1, 0))  # (L,N)
            embeddings = tf.get_variable(
                dtype=tf.float32,
                name="embeddings",
                shape=[vocab_size, params.encoder_dim],
                initializer=tf.initializers.truncated_normal(
                    stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
            h = tf.nn.embedding_lookup(embeddings, h)  # (L, N, D)
            ls = tf.linspace(
                start=tf.constant(0, dtype=tf.float32),
                stop=tf.constant(1, dtype=tf.float32),
                num=L)  # (L,)
            ls = tf.tile(tf.expand_dims(ls, 1), [1, n])  # (L,N)
            ls = ls * tf.cast(L, dtype=tf.float32) / tf.cast(tf.expand_dims(token_lengths, 0), dtype=tf.float32)
            ls = tf.expand_dims(ls, 2)  # ( L,N,1)
            h = tf.concat([h, ls], axis=-1)
            hidden_state, hidden_state_final = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True,
                sequence_lengths=token_lengths
            )
            h = tf.concat(hidden_state_final, axis=-1)  # (layers*directions, N, D)
            h = tf.transpose(h, (1, 0, 2))  # (N,layers*directions,D)
            h = tf.reshape(h, (n, h.shape[1].value * h.shape[2].value))  # (N, layers*directions*D)
            h = slim.batch_norm(inputs=h, is_training=True)
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_1',
                weights_regularizer=weights_regularizer
            )
            h = slim.batch_norm(inputs=h, is_training=True)
            """
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_2',
                weights_regularizer=weights_regularizer
            )
            """
            flat_encoding = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_3',
                weights_regularizer=weights_regularizer
            )  # (N,D)
        with tf.variable_scope('step_2'):
            h = tf.expand_dims(flat_encoding, axis=0)  # (1, N, D)
            h = tf.tile(h, (output_length, 1, 1))  # (O,N,D)
            ls = tf.linspace(start=-1., stop=1., num=params.flat_length)  # (O,)
            ls = tf.tile(tf.expand_dims(tf.expand_dims(ls, 1), 2), (1, n, 1))  # (O,N,1)
            h = tf.concat([h, ls], axis=2)
            output_hidden, _ = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True
            )  # (O, N, D)
            # output_hidden = sequence_norm(output_hidden)
            output_hidden = slim.batch_norm(inputs=output_hidden, is_training=is_training)
        with tf.variable_scope('encoder_attn'):
            output_proj = slim.fully_connected(
                inputs=output_hidden,
                num_outputs=params.attention_dim,
                activation_fn=None,
                scope='encoder_output_proj',
                weights_regularizer=weights_regularizer
            )  # (O,N,D)
            input_proj = slim.fully_connected(
                inputs=hidden_state,
                num_outputs=params.attention_dim,
                activation_fn=None,
                scope='encoder_input_proj',
                weights_regularizer=weights_regularizer
            )  # (O,N,D)
            attn = calc_attn_v2(output_proj, input_proj, token_lengths)  # (n, ol, il)
            tf.summary.image('encoder_attention', tf.expand_dims(attn, 3))
            input_aligned = tf.matmul(
                attn,  # (n, ol, il)
                tf.transpose(hidden_state, (1, 0, 2))  # (n, il, d)
            )  # (n, ol, d)
            h = tf.concat([tf.transpose(input_aligned, (1, 0, 2)), output_hidden], axis=-1)
        with tf.variable_scope('encoder_output'):
            # h = sequence_norm(h)
            h = slim.batch_norm(h, is_training=is_training)
            h, _ = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True
            )  # (O, N, D)
            """
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=None,
                scope='encoder_mlp_out_1',
                weights_regularizer=weights_regularizer
            )
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=None,
                scope='encoder_mlp_out_2',
                weights_regularizer=weights_regularizer
            )
            """
            # h = sequence_norm(h)
            h = slim.batch_norm(h, is_training=is_training)
            mu = slim.fully_connected(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='encoder_mlp_mu',
                weights_regularizer=weights_regularizer
            )
            logsigma = slim.fully_connected(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='encoder_mlp_logsigma',
                weights_regularizer=weights_regularizer
            )
            return mu, logsigma


def vae_flat_decoder_attn(latent, vocab_size, params, n, weights_regularizer=None, is_training=True):
    # latent (N, D)
    with tf.variable_scope('decoder'):
        """
        h = slim.fully_connected(
            latent,
            num_outputs=params.decoder_dim,
            scope='projection',
            activation_fn=None,
            weights_regularizer=weights_regularizer
        )
        """
        h = latent
        # h = sequence_norm(h)
        h = slim.batch_norm(h, is_training=is_training)
        h, _ = lstm(
            x=h,
            num_layers=3,
            num_units=params.decoder_dim,
            bidirectional=True
        )
        # h = sequence_norm(h)
        h = slim.batch_norm(h, is_training=is_training)
        """
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='decoder_mlp_1',
            weights_regularizer=weights_regularizer
        )
        h = slim.fully_connected(
            inputs=h,
            num_outputs=params.encoder_dim,
            activation_fn=tf.nn.leaky_relu,
            scope='decoder_mlp_2',
            weights_regularizer=weights_regularizer
        )
        """
        h = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='decoder_mlp_3',
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.initializers.constant(
                value=get_bias_ctc(average_output_length=params.flat_length, smoothing=params.bias_smoothing),
                verify_shape=True)
        )  # (L,N,V+1)
        return h


def make_model_vae_ctc_flat_attn(
        run_config,
        vocab,
        merge_repeated=False
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

        with tf.control_dependencies([
            tf.assert_greater_equal(params.flat_length, token_lengths, message="Tokens longer than tree size"),
            tf.assert_greater(vocab_size, tokens, message="Tokens larger than vocab"),
            tf.assert_greater_equal(tokens, 0, message="Tokens less than 0")
        ]):
            tokens = tf.identity(tokens)

        if params.l2 > 0:
            weights_regularizer = slim.l2_regularizer(params.l2)
        else:
            weights_regularizer = None

        # Encoder
        mu, logsigma = vae_flat_encoder_attn(
            tokens=tokens,
            token_lengths=token_lengths,
            vocab_size=vocab_size,
            params=params,
            n=n,
            weights_regularizer=weights_regularizer,
            output_length=params.flat_length,
            is_training=is_training
        )
        # Sampling
        latent_sample, latent_prior_sample = kl(
            mu=mu,
            logsigma=logsigma,
            params=params,
            n=n)  # (L,N,D)

        # Decoder
        with tf.variable_scope('vae_decoder') as decoder_scope:
            logits = vae_flat_decoder_attn(
                latent=latent_sample,
                vocab_size=vocab_size,
                params=params,
                weights_regularizer=weights_regularizer,
                n=n,
                is_training=is_training
            )
            sequence_length_ctc = tf.tile([params.flat_length], (n,))  # tf.shape(logits)[0:1], (n,))

        ctc_labels_sparse = sparsify(tokens, sequence_mask)
        ctc_labels = tf.sparse_tensor_to_dense(ctc_labels_sparse, default_value=-1)
        print("Labels: {}".format(ctc_labels))
        print("CTC: {}, {}, {}".format(ctc_labels, logits, sequence_length_ctc))
        ctc_loss_raw = tf.nn.ctc_loss(
            labels=ctc_labels_sparse,
            sequence_length=sequence_length_ctc,
            inputs=logits,
            # sequence_length=tf.shape(logits)[0],
            ctc_merge_repeated=merge_repeated,
            # preprocess_collapse_repeated=False,
            # ctc_merge_repeated=True,
            # ignore_longer_outputs_than_inputs=False,
            time_major=True
        )
        ctc_loss = tf.reduce_mean(ctc_loss_raw)
        tf.losses.add_loss(ctc_loss)

        total_loss = tf.losses.get_total_loss()

        # Generated data
        with tf.variable_scope(decoder_scope, reuse=True):
            glogits = vae_flat_decoder_attn(
                latent=latent_prior_sample,
                vocab_size=vocab_size,
                params=params,
                weights_regularizer=weights_regularizer,
                n=n
            )

        autoencode_hook = CTCHook(
            logits=logits,
            lengths=sequence_length_ctc,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "autoencoded", "autoencoded-{:08d}.csv"),
            true=ctc_labels,
            name="Autoencoded",
            merge_repeated=merge_repeated
        )
        generate_hook = CTCHook(
            logits=glogits,
            lengths=sequence_length_ctc,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv"),
            true=ctc_labels,
            name="Generated",
            merge_repeated=merge_repeated
        )
        evaluation_hooks = [autoencode_hook, generate_hook]

        tf.summary.scalar('ctc_loss', ctc_loss)
        tf.summary.scalar('total_loss', total_loss)

        # Train
        optimizer = tf.train.AdamOptimizer(params.lr)
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            clip_gradient_norm=params.clip_gradient_norm)
        eval_metric_ops = {
            'ctc_loss_eval': tf.metrics.mean(ctc_loss_raw),
            'token_lengths_eval': tf.metrics.mean(token_lengths)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=evaluation_hooks,
            train_op=train_op)

    return model_fn
