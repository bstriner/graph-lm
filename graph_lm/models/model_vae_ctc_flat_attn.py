import os

import tensorflow as tf
from tensorflow.contrib import slim

from .attn_util import calc_attn
from .rnn_util import lstm
from ..callbacks.ctc_callback import CTCHook
from ..kl import kl
from ..sparse import sparsify


def vae_flat_encoder_attn(tokens, token_lengths, vocab_size, params, n, output_length, weights_regularizer=None):
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
            hidden_state, hidden_state_final = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True,
                sequence_lengths=token_lengths
            )
            h = tf.concat(hidden_state_final, axis=-1)
            h = tf.transpose(h, (1, 0, 2))  # (N,2,D)
            h = tf.reshape(h, (n, h.shape[1].value * h.shape[2].value))  # (N, 2D)
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_1',
                weights_regularizer=weights_regularizer
            )
            h = slim.fully_connected(
                inputs=h,
                num_outputs=params.encoder_dim,
                activation_fn=tf.nn.leaky_relu,
                scope='encoder_mlp_2',
                weights_regularizer=weights_regularizer
            )
            flat_encoding = slim.fully_connected(
                inputs=h,
                num_outputs=params.latent_dim,
                activation_fn=None,
                scope='encoder_mlp_3',
                weights_regularizer=weights_regularizer
            )
        with tf.variable_scope('step_2'):
            h = tf.expand_dims(flat_encoding, axis=0)  # (1, N, D)
            h = tf.tile(h, (output_length, 1, 1))  # (L,N,D)
            output_hidden, _ = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True
            )
        with tf.variable_scope('encoder_attn'):
            output_proj = slim.fully_connected(
                inputs=output_hidden,
                num_outputs=params.attention_dim,
                activation_fn=None,
                scope='encoder_output_proj',
                weights_regularizer=weights_regularizer
            )
            input_proj = slim.fully_connected(
                inputs=hidden_state,
                num_outputs=params.attention_dim,
                activation_fn=None,
                scope='encoder_input_proj',
                weights_regularizer=weights_regularizer
            )
            attn = calc_attn(output_proj, input_proj, token_lengths)  # (n, ol, il)
            tf.summary.image('encoder_attention', tf.expand_dims(attn, 3))
            input_aligned = tf.matmul(
                attn,  # (n, ol, il)
                tf.transpose(hidden_state, (1, 0, 2))  # (n, il, d)
            )  # (n, ol, d)
            h = tf.concat([tf.transpose(input_aligned, (1, 0, 2)), output_hidden], axis=-1)
        with tf.variable_scope('encoder_output'):
            h, _ = lstm(
                x=h,
                num_layers=2,
                num_units=params.encoder_dim,
                bidirectional=True
            )
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


def vae_flat_decoder_attn(latent, vocab_size, params, n, weights_regularizer=None):
    # latent (N, D)
    with tf.variable_scope('decoder'):
        depth = params.tree_depth
        assert depth >= 0
        h = slim.fully_connected(
            latent,
            num_outputs=params.decoder_dim,
            scope='projection',
            activation_fn=None,
            weights_regularizer=weights_regularizer
        )
        h, _ = lstm(
            x=h,
            num_layers=2,
            num_units=params.decoder_dim,
            bidirectional=True
        )
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
            weights_regularizer=weights_regularizer
        )  # (L,N,V+1)
        return h


def make_model_vae_ctc_flat_attn(
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
            output_length=params.flat_length
        )
        # Sampling
        latent_sample, latent_prior_sample = kl(
            mu=mu,
            logsigma=logsigma,
            params=params,
            n=n)

        # Decoder
        with tf.variable_scope('vae_decoder') as decoder_scope:
            logits = vae_flat_decoder_attn(
                latent=latent_sample,
                vocab_size=vocab_size,
                params=params,
                weights_regularizer=weights_regularizer,
                n=n
            )
            sequence_length_ctc = tf.tile(tf.shape(logits)[0:1], (n,))

        ctc_labels_sparse = sparsify(tokens, sequence_mask)
        ctc_labels = tf.sparse_tensor_to_dense(ctc_labels_sparse, default_value=-1)
        # ctc_labels = tf.sparse_transpose(ctc_labels, (1,0))
        print("Labels: {}".format(ctc_labels))
        # tf.tile(tf.pow([2], depth), (n,))
        print("CTC: {}, {}, {}".format(ctc_labels, logits, sequence_length_ctc))
        ctc_loss_raw = tf.nn.ctc_loss(
            labels=ctc_labels_sparse,
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
            name="Autoencoded"
        )
        generate_hook = CTCHook(
            logits=glogits,
            lengths=sequence_length_ctc,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv"),
            true=ctc_labels,
            name="Generated"
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
