import os

import tensorflow as tf
from tensorflow.contrib import slim

from .networks.decoder_flat_attn import vae_flat_decoder_attn
from .networks.encoder_flat_attn import vae_flat_encoder_attn
from ..callbacks.ctc_callback import CTCHook
from ..kl import kl
from ..sparse import sparsify


def make_model_vae_ctc_flat_attn(
        run_config,
        vocab,
        merge_repeated=True
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
