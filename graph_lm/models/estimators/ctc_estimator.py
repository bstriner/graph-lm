import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.estimator.estimator_lib import EstimatorSpec

from ...callbacks.ctc_callback import CTCHook
from ...sparse import sparsify


def ctc_estimator(
        tokens, token_lengths, logits, glogits,
        sequence_mask, sequence_length_ctc, vocab, run_config, params, mode,
        model_scope,
        training_hooks=[]):

    with tf.name_scope(model_scope+"/"):
        ctc_labels_sparse = sparsify(tf.cast(tokens, tf.int32), sequence_mask)
        ctc_labels = tf.sparse_tensor_to_dense(ctc_labels_sparse, default_value=-1)
        # ctc_labels = tf.sparse_transpose(ctc_labels, (1,0))
        print("Labels: {}".format(ctc_labels))
        print("logits: {}".format(logits))
        print("glogits: {}".format(glogits))
        # tf.tile(tf.pow([2], depth), (n,))
        print("CTC: {}, {}, {}".format(ctc_labels, logits, sequence_length_ctc))
        ctc_loss_raw = tf.nn.ctc_loss_v2(
            labels=tokens,
            label_length=token_lengths,
            logits=logits,
            logit_length=sequence_length_ctc,
            blank_index=-1
            # sequence_length=tf.shape(logits)[0],
            # ctc_merge_repeated=True,
            # preprocess_collapse_repeated=False,
            # ctc_merge_repeated=True,
            # ignore_longer_outputs_than_inputs=False,
            # time_major=True
        )
        ctc_loss = tf.reduce_mean(ctc_loss_raw, name='ctc_loss')
        tf.losses.add_loss(ctc_loss)

    losses = tf.losses.get_losses(scope=model_scope)
    print("Estimator losses: {}".format(losses))
    losses += tf.losses.get_regularization_losses(scope=model_scope)
    total_loss = tf.add_n(losses)
    updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=model_scope)

    autoencode_hook = CTCHook(
        logits=logits,
        lengths=sequence_length_ctc,
        vocab=vocab,
        path=os.path.join(run_config.model_dir, "autoencoded", "autoencoded-{:08d}.csv"),
        true=ctc_labels,
        name="Autoencoded",
        merge_repeated=True
    )
    generate_hook = CTCHook(
        logits=glogits,
        lengths=sequence_length_ctc,
        vocab=vocab,
        path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv"),
        true=ctc_labels,
        name="Generated",
        merge_repeated=True
    )
    evaluation_hooks = [autoencode_hook, generate_hook]

    tf.summary.scalar('ctc_loss', ctc_loss)
    tf.summary.scalar('total_loss', total_loss)

    # Train
    optimizer = tf.train.AdamOptimizer(params.lr)
    variables = tf.trainable_variables(scope=model_scope)
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        clip_gradient_norm=params.clip_gradient_norm,
        variables_to_train=variables,
        update_ops=updates)
    eval_metric_ops = {
        'ctc_loss_eval': tf.metrics.mean(ctc_loss_raw),
        'token_lengths_eval': tf.metrics.mean(token_lengths)
    }

    return EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks,
        training_hooks=training_hooks,
        train_op=train_op)
