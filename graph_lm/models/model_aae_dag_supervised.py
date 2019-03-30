import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from .networks.dag_utils import build_dag
from .networks.decoder_dag_supervised import vae_decoder_dag_supervised
from .networks.discriminator_dag_supervised import discriminator_dag_supervised
from .networks.encoder_dag_gan import encoder_dag_gan
from ..anneal import get_kl_scale_logistic
from ..callbacks.dag_callback import DAGHook


def make_model_aae_dag_supervised(
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
        noise = tf.random_normal(
            shape=(n, L, params.noise_dim),
            dtype=tf.float32
        )
        idx = tf.where(sequence_mask)

        with tf.variable_scope("aae") as aae_scope:
            # Encoder
            with tf.variable_scope("encoder_scope") as encoder_scope:
                encoded = encoder_dag_gan(
                    dag_bw=dag_bw,
                    dag=dag,
                    noise=noise,
                    text=text,
                    tags=tags,
                    vocab_size=vocab_size,
                    tags_size=tag_size,
                    params=params,
                    weights_regularizer=weights_regularizer
                )  # (N,L,D)
            with tf.variable_scope("decoder_scope") as decoder_scope:
                decoded = vae_decoder_dag_supervised(
                    latent=encoded,
                    vocab_size=vocab_size,
                    dag=dag,
                    dag_bw=dag_bw,
                    tags=tags,
                    tag_size=tag_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training
                )
        with tf.variable_scope("discriminator") as  discriminator_scope:
            with tf.name_scope("fake"):
                y_fake = discriminator_dag_supervised(
                    latent=encoded,
                    dag=dag,
                    dag_bw=dag_bw,
                    tags=tags,
                    tag_size=tag_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training,
                    idx=idx
                )  # (N,)
        with tf.variable_scope(discriminator_scope, reuse=True):
            with tf.name_scope("real"):
                latent_prior = tf.random_normal(
                    shape=(n, L, params.latent_dim),
                    dtype=tf.float32,
                    name="latent_prior"
                )
                y_real = discriminator_dag_supervised(
                    latent=latent_prior,
                    dag=dag,
                    dag_bw=dag_bw,
                    tags=tags,
                    tag_size=tag_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training,
                    idx=idx
                )  # (N,)

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
        tf.summary.scalar("wgan_loss", wgan_loss)
        wgan_loss_scaled = wgan_loss*wgan_scale
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
