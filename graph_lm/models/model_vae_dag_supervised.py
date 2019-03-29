import os

import tensorflow as tf
from tensorflow.contrib import slim

from .networks.dag_utils import build_dag
from .networks.decoder_dag_supervised import vae_decoder_dag_supervised
from .networks.encoder_dag import encoder_dag
from ..callbacks.dag_callback import DAGHook
from ..kl import kl


def make_model_vae_dag_supervised(
        run_config,
        vocab,
        taglist
):
    vocab_size = vocab.shape[0]
    tag_size = taglist.shape[0]
    print("Vocab size: {}".format(vocab_size))

    def model_fn(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        # Inputs
        sequence_length = features['sequence_length']
        text = features['text']
        tags = features['tags']
        heads = features['heads']
        sequence_mask = tf.sequence_mask(maxlen=tf.shape(text)[1], lengths=sequence_length)  # (N,L)
        n = tf.shape(text)[0]
        L = tf.shape(text)[1]

        with tf.control_dependencies([
            tf.assert_greater(tf.constant(vocab_size, dtype=text.dtype), text, message="Tokens larger than vocab"),
            tf.assert_greater_equal(text, tf.constant(0, dtype=text.dtype), message="Tokens less than 0")
        ]):
            text = tf.identity(text)

        if params.l2 > 0:
            weights_regularizer = slim.l2_regularizer(params.l2)
        else:
            weights_regularizer = None

        dag = build_dag(heads=heads)  # (n, dep, head)
        dag_bw = tf.transpose(dag, (0, 2, 1))  # (n, head, dep)

        # Encoder
        mu, logsigma = encoder_dag(
            dag_bw=dag_bw,
            dag=dag,
            text=text,
            tags=tags,
            vocab_size=vocab_size,
            tags_size=tag_size,
            params=params,
            weights_regularizer=weights_regularizer
        )  # (N,L,D)

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
            logits = vae_decoder_dag_supervised(
                latent=latent_sample,
                vocab_size=vocab_size,
                dag=dag,
                dag_bw=dag_bw,
                tags=tags,
                tag_size=tag_size,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training
            )

        # Loss calculation
        logits_values = tf.gather_nd(params=logits, indices=idx)
        labels_values = tf.gather_nd(params=text, indices=idx)
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
            glogits = vae_decoder_dag_supervised(
                latent=latent_prior_sample,
                vocab_size=vocab_size,
                dag=dag,
                dag_bw=dag_bw,
                tags=tags,
                tag_size=tag_size,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training
            )
        # Hooks
        autoencode_hook = DAGHook(
            logits=logits,
            true=text,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "autoencoded", "autoencoded-{:08d}.csv"),
            name="Autoencoded",
            idx=idx
        )
        generate_hook = DAGHook(
            logits=glogits,
            true=text,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv"),
            name="Generated",
            idx=idx
        )
        evaluation_hooks = [autoencode_hook, generate_hook]

        # tf.summary.scalar('model_total_loss', total_loss)

        # Train
        optimizer = tf.train.AdamOptimizer(params.lr)
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            clip_gradient_norm=params.clip_gradient_norm)
        eval_metric_ops = {
            'cross_entropy_eval': tf.metrics.mean(loss_n),
            'token_lengths_eval': tf.metrics.mean(sequence_length)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=evaluation_hooks,
            train_op=train_op)

    return model_fn
