import tensorflow as tf
from tensorflow.contrib import slim

from .asr_vae import asr_vae_train, asr_vae_gen
from .callbacks import AsrHook

PAD_TRANSCRIPTS = 60
GEN_TRANSCRIPTS = 300


def make_model_fn(
        run_config,
        vocab
):
    vocab_size = vocab.shape[0]

    def model_fn(features, labels, mode, params):
        utterances = features['utterances']  # (N, L, 64)
        utterance_lengths = features['utterance_lengths']  # (N,)
        utterance_masks = tf.sequence_mask(
            utterance_lengths, maxlen=tf.shape(utterances)[1], name='utterance_masks'
        )  # (N, L)
        tf.summary.image("mel_input", tf.expand_dims(utterances, 3))

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            transcripts = features['transcripts']  # (N, T)
            transcript_lengths = features['transcript_lengths']
            if PAD_TRANSCRIPTS > 0:
                transcripts = tf.pad(transcripts, [(0, 0), (0, PAD_TRANSCRIPTS)])
            transcript_masks = tf.sequence_mask(
                transcript_lengths, maxlen=tf.shape(transcripts)[1], name='utterance_masks'
            )  # (N, T)

            with tf.variable_scope('asr_vae') as vs:
                logits = asr_vae_train(
                    params=params,
                    vocab_size=vocab_size,
                    utterances=utterances,
                    utterance_mask=utterance_masks,
                    transcripts=transcripts,
                    transcript_mask=transcript_masks)  # (N, T, V)
            with tf.variable_scope(vs, reuse=True):
                gen = asr_vae_gen(
                    params=params,
                    vocab_size=vocab_size,
                    utterances=utterances,
                    utterance_mask=utterance_masks,
                    t=GEN_TRANSCRIPTS
                )

            # Loss
            one_hot = tf.one_hot(indices=transcripts, depth=vocab_size + 1, axis=2)
            softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=one_hot,
                logits=logits,
                dim=2)  # (N, T)
            masked_loss = tf.reduce_mean(tf.reduce_sum(softmax_loss * tf.cast(transcript_masks, tf.float32), 1), 0)
            tf.losses.add_loss(masked_loss, loss_collection=tf.GraphKeys.LOSSES)
            if params.l2 > 0:
                with tf.name_scope("l2calculation"):
                    for v in tf.trainable_variables():
                        tf.losses.add_loss(
                            tf.nn.l2_loss(v) * params.l2,
                            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES
                        )

            reg_loss = tf.losses.get_regularization_loss()
            total_loss = tf.losses.get_total_loss()

            kls = tf.get_collection('KL')
            assert len(kls) == 1
            kl = kls[0]
            tf.summary.scalar('softmax_loss', masked_loss)
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('kl_loss', kl)

            # Train
            optimizer = tf.train.AdamOptimizer(params.lr)
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)
            else:
                eval_metric_ops = {}
                asr_hook = AsrHook(
                    true=transcripts,
                    generated=gen,
                    vocab=vocab,
                    path=run_config.model_dir)
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=total_loss,
                    eval_metric_ops=eval_metric_ops,
                    evaluation_hooks=[asr_hook])
        else:
            raise NotImplementedError()

    return model_fn
