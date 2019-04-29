import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.estimator.estimator_lib import EstimatorSpec

from ...callbacks.dag_callback import DAGHook


def crossentropy_estimator(
        tokens, token_lengths, logits, glogits, vocab, run_config, params, mode,
        model_scope, idx,
        training_hooks=[]):
    n = tf.shape(tokens)[0]
    L = tf.shape(tokens)[1]
    # Loss calculation

    with tf.name_scope(model_scope + "/loss/"):
        logits_values = tf.gather_nd(params=logits, indices=idx)
        labels_values = tf.gather_nd(params=tokens, indices=idx)
        onehot_labels_values = tf.one_hot(indices=labels_values, depth=vocab.shape[0])
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

    losses = tf.losses.get_losses(scope=model_scope)
    print("Estimator losses: {}".format(losses))
    losses += tf.losses.get_regularization_losses(scope=model_scope)
    total_loss = tf.add_n(losses)
    updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=model_scope)

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

    # tf.summary.scalar('model_total_loss', total_loss)

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
        'cross_entropy_eval': tf.metrics.mean(loss_n),
        'token_lengths_eval': tf.metrics.mean(token_lengths)
    }

    return EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks,
        training_hooks=training_hooks,
        train_op=train_op)
