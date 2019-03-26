import os

import tensorflow as tf
from tensorflow.contrib import slim

from .networks.decoder_dag import vae_decoder_dag
from .networks.encoder_simple import vae_flat_encoder_simple
from ..anneal import get_penalty_scale_logistic
from ..callbacks.dag_callback import DAGHook
from ..kl import kl


def make_model_vae_dag(
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
        sequence_mask = tf.sequence_mask(maxlen=tf.shape(tokens)[1], lengths=token_lengths)  # (N,L)
        n = tf.shape(tokens)[0]
        L = tf.shape(tokens)[1]

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
        mu_t, logsigma_t = vae_flat_encoder_simple(
            tokens=tokens,
            token_lengths=token_lengths,
            vocab_size=vocab_size,
            params=params,
            n=n,
            weights_regularizer=weights_regularizer
        )  # (L,N,D)
        mu = tf.transpose(mu_t, (1, 0, 2))  # (N,L,D)
        logsigma = tf.transpose(logsigma_t, (1, 0, 2))  # (N,L,D)

        # Sampling
        idx = tf.where(sequence_mask)
        with tf.name_scope("kl"):
            selected_mu = tf.gather_nd(params=mu, indices=idx)
            selected_logsigma = tf.gather_nd(params=logsigma, indices=idx)
            latent_sample_values, latent_prior_sample_values = kl(
                mu=selected_mu,
                logsigma=selected_logsigma,
                params=params,
                n=n)
            latent_sample = tf.scatter_nd(
                updates=latent_sample_values,
                indices=idx,
                shape=(n, L, latent_sample_values.shape[-1].value)
            )  # (N,L,D)
            latent_prior_sample = tf.scatter_nd(
                updates=latent_prior_sample_values,
                indices=idx,
                shape=(n, L, latent_prior_sample_values.shape[-1].value)
            )  # (N,L,D)

        # Decoder
        with tf.variable_scope('vae_decoder') as decoder_scope:
            logits, penalty = vae_decoder_dag(
                latent=latent_sample,
                vocab_size=vocab_size,
                sequence_lengths=token_lengths,
                params=params,
                weights_regularizer=weights_regularizer,
                n=n,
                is_training=is_training
            )
        with tf.name_scope("dag_penalty"):
            penalty_scale = get_penalty_scale_logistic(params)
            dag_penalty_raw = tf.reduce_mean(tf.square(penalty))
            weighted_dag_penalty = penalty_scale * dag_penalty_raw
            tf.losses.add_loss(loss=weighted_dag_penalty, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            tf.summary.scalar('dag_penalty_scale', penalty_scale)
            tf.summary.scalar('dag_penalty_raw', dag_penalty_raw)
            tf.summary.scalar('dag_penalty_weighted', weighted_dag_penalty)

        # Loss calculation
        logits_values = tf.gather_nd(params=logits, indices=idx)
        labels_values = tf.gather_nd(params=tokens, indices=idx)
        onehot_labels_values = tf.one_hot(indices=labels_values, depth=vocab_size)
        loss_values = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels_values,
            logits=logits_values,
            reduction=tf.losses.Reduction.NONE,
            loss_collection=None
        )
        loss_arr = tf.scatter_nd(updates=loss_values, indices=idx, shape=(n, L))
        loss_n = tf.reduce_sum(loss_arr, axis=-1)
        loss = tf.reduce_mean(loss_n)
        tf.losses.add_loss(loss)
        tf.summary.scalar("softmax_cross_entropy", loss)

        total_loss = tf.losses.get_total_loss()

        # Generated data
        with tf.variable_scope(decoder_scope, reuse=True):
            glogits, _ = vae_decoder_dag(
                latent=latent_prior_sample,
                vocab_size=vocab_size,
                sequence_lengths=token_lengths,
                params=params,
                weights_regularizer=weights_regularizer,
                n=n,
                is_training=is_training
            )

        # Hooks
        autoencode_hook = DAGHook(
            logits=logits,
            true=tokens,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "autoencoded", "autoencoded-{:08d}.csv"),
            name="Autoencoded",
            idx=idx
        )
        generate_hook = DAGHook(
            logits=glogits,
            true=tokens,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv"),
            name="Generated",
            idx=idx
        )
        evaluation_hooks = [autoencode_hook, generate_hook]

        #tf.summary.scalar('model_total_loss', total_loss)

        # Train
        optimizer = tf.train.AdamOptimizer(params.lr)
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            clip_gradient_norm=params.clip_gradient_norm)
        eval_metric_ops = {
            'cross_entropy_eval': tf.metrics.mean(loss_n),
            'token_lengths_eval': tf.metrics.mean(token_lengths)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=evaluation_hooks,
            train_op=train_op)

    return model_fn
