import os

import tensorflow as tf
from tensorflow.contrib import slim

from ...callbacks.ctc_callback import CTCHook
from ...sparse import sparsify


def aae_ctc_estimator(
        tokens, token_lengths, logits, glogits, idx,
        sequence_mask, sequence_length_ctc, vocab, run_config, params, mode):
    ctc_labels_sparse = sparsify(tf.cast(tokens, tf.int32), sequence_mask)
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
        # ctc_merge_repeated=False,
        # preprocess_collapse_repeated=False,
        # ctc_merge_repeated=True,
        # ignore_longer_outputs_than_inputs=False,
        time_major=True
    )
    ctc_loss = tf.reduce_mean(ctc_loss_raw)
    tf.losses.add_loss(ctc_loss)



    wgan_loss_n = y_real - y_fake
    wgan_loss = tf.reduce_mean(wgan_loss_n)
    wgan_scale = get_kl_scale_logistic(params=params)
    wgan_loss_scaled = wgan_loss * wgan_scale
    tf.summary.scalar("wgan_loss", wgan_loss)
    tf.summary.scalar("wgan_scale", wgan_scale)
    tf.summary.scalar("wgan_loss_scaled", wgan_loss_scaled)

    aae_reg = tf.losses.get_regularization_loss(scope=aae_scope.name)
    dis_reg = tf.losses.get_regularization_loss(scope=discriminator_scope.name)

    aae_loss = aae_reg + softmax_cross_entropy - wgan_loss_scaled
    dis_loss = dis_reg + wgan_loss

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks,
        train_op=train_op)
