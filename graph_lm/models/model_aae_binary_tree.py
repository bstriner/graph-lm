import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .networks.decoder_dag_supervised import vae_decoder_dag_supervised
from ..anneal import get_kl_scale_logistic
from ..callbacks.dag_callback import DAGHook

from .networks.decoder_bintree_attention import decoder_bintree_attention
from .networks.encoder_bintree_recurrent_attention import encoder_bintree_recurrent_attn_aae
from ..data.word import SENTENCE_LENGTH, TEXT

import math
def make_model_aae_binary_tree(
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
        tokens = features[TEXT]  # (N, L)
        token_lengths = features[SENTENCE_LENGTH]  # (N,)
        sequence_mask = tf.sequence_mask(maxlen=tf.shape(tokens)[1], lengths=token_lengths)
        n = tf.shape(tokens)[0]
        depth = params.tree_depth
        const_sequence_length = int(math.pow(2, depth + 1) - 1)

        # Assertions
        with tf.control_dependencies([
            tf.assert_greater_equal(
                tf.cast(const_sequence_length, dtype=token_lengths.dtype),
                token_lengths,
                message="Tokens longer than tree size"),
            tf.assert_less(
                tokens,
                tf.cast(vocab_size, tokens.dtype),
                message="Tokens larger than vocab"),
            tf.assert_greater_equal(
                tokens,
                tf.constant(0, dtype=tokens.dtype),
                message="Tokens less than 0")
        ]):
            tokens = tf.identity(tokens)

        if params.l2 > 0:
            weights_regularizer = slim.l2_regularizer(params.l2)
        else:
            weights_regularizer = None

        # Encoder
        noise = tf.random_normal(
            shape=(tf.shape(tokens)[0], tf.shape(tokens)[1], params.noise_dim)
        )
        encoding = encoder_bintree_recurrent_attn_aae(
            tokens=tokens,
            noise=noise,
            token_lengths=token_lengths,
            vocab_size=vocab_size,
            params=params,
            n=n,
            weights_regularizer=weights_regularizer,
            is_training=is_training
        )  # [(N,1,D), (N,2,D),(N,4,D),...]
        encoding_prior = [tf.random_normal(
            shape=(tf.shape(enc)[0], tf.shape(enc)[1], enc.shape[2].value)
        ) for enc in encoding]
        # Decoder
        with tf.variable_scope('vae_decoder') as decoder_scope:
            logits = decoder_bintree_attention(
                latent_layers=encoding,
                vocab_size=vocab_size,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training)  # (L,N,D)
        # Generated data
        with tf.variable_scope(decoder_scope, reuse=True):
            glogits = decoder_bintree_attention(
                latent_layers=encoding_prior,
                vocab_size=vocab_size,
                params=params,
                weights_regularizer=weights_regularizer,
                is_training=is_training)  # (L,N,D)
        with tf.control_dependencies([
            tf.assert_equal(
                tf.cast(const_sequence_length, dtype=token_lengths.dtype),
                tf.shape(logits)[0],
                message='tree output shape incorrect')
        ]):
            logits = tf.identity(logits)
        sequence_length_ctc = tf.tile(tf.constant([int(math.pow(2, depth + 1) - 1)], dtype=tf.int32), (n,))

        # Loss calculation
        with tf.name_scope("loss"):
            logits_values = tf.gather_nd(params=decoded, indices=idx)
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

        softmax_cross_entropy = tf.reduce_mean(loss_n)
        tf.losses.add_loss(softmax_cross_entropy)
        tf.summary.scalar("softmax_cross_entropy", softmax_cross_entropy)
        wgan_loss_n = y_real - y_fake
        wgan_loss = tf.reduce_mean(wgan_loss_n)
        wgan_scale = get_kl_scale_logistic(params=params)
        wgan_loss_scaled = wgan_loss*wgan_scale
        tf.summary.scalar("wgan_loss", wgan_loss)
        tf.summary.scalar("wgan_scale", wgan_scale)
        tf.summary.scalar("wgan_loss_scaled", wgan_loss_scaled)

        aae_reg = tf.losses.get_regularization_loss(scope=aae_scope.name)
        dis_reg = tf.losses.get_regularization_loss(scope=discriminator_scope.name)

        aae_loss = aae_reg + softmax_cross_entropy - wgan_loss_scaled
        dis_loss = dis_reg + wgan_loss

        # Generated data
        with tf.variable_scope(decoder_scope, reuse=True):
            encoded_prior = tf.random_normal(
                shape=tf.shape(encoded),
                dtype=tf.float32
            )
            decoded_gen = vae_decoder_dag_supervised(
                latent=encoded_prior,
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
            logits=decoded,
            true=text,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "autoencoded", "autoencoded-{:08d}.csv"),
            name="Autoencoded",
            idx=idx
        )
        generate_hook = DAGHook(
            logits=decoded_gen,
            true=text,
            vocab=vocab,
            path=os.path.join(run_config.model_dir, "generated", "generated-{:08d}.csv"),
            name="Generated",
            idx=idx
        )
        # Train
        updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=aae_scope.name)
        dis_updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=discriminator_scope.name)
        optimizer = tf.train.AdamOptimizer(params.lr)
        train_op = slim.learning.create_train_op(
            aae_loss,
            optimizer,
            clip_gradient_norm=params.clip_gradient_norm,
            variables_to_train=tf.trainable_variables(scope=aae_scope.name),
            update_ops=updates)

        dis_optimizer = tf.train.AdamOptimizer(params.dis_lr)
        dis_train_op = slim.learning.create_train_op(
            dis_loss,
            dis_optimizer,
            clip_gradient_norm=params.clip_gradient_norm,
            variables_to_train=tf.trainable_variables(scope=discriminator_scope.name),
            update_ops=dis_updates)

        eval_metric_ops = {
            'cross_entropy_eval': tf.metrics.mean(loss_n),
            'token_lengths_eval': tf.metrics.mean(sequence_length)
        }
        discriminator_hook = RunTrainOpsHook(
            train_ops=[dis_train_op],
            train_steps=params.discriminator_steps
        )
        training_hooks = [discriminator_hook]
        evaluation_hooks = [autoencode_hook, generate_hook]

        # tf.summary.scalar('model_total_loss', total_loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=aae_loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=evaluation_hooks,
            training_hooks=training_hooks,
            train_op=train_op)

    return model_fn
