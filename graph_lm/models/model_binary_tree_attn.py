import math
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.gan.python.train import RunTrainOpsHook

from graph_lm.models.estimators.kl import kl_array
from .estimators.aae import sample_aae_array
from .estimators.vae_ctc_estimator import ctc_estimator
from .networks.decoder_bintree_attention import decoder_bintree_attention
from .networks.discriminator_bintree import discriminator_bintree_fn
from .networks.encoder_bintree_recurrent_attention import encoder_bintree_recurrent_attn_vae
from .networks.utils.bintree_utils import concat_layers
from ..anneal import get_penalty_scale_logistic
from ..data.word import SENTENCE_LENGTH, TEXT


def make_model_binary_tree_attn(
        run_config,
        vocabs,
        model_mode="vae"
):
    vocab = vocabs[TEXT]
    vocab_size = vocab.shape[0]
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
        with tf.variable_scope("autoencoder") as autoencoder_scope:
            with tf.variable_scope('encoder'):
                mus, logsigmas = encoder_bintree_recurrent_attn_vae(
                    tokens=tokens,
                    token_lengths=token_lengths,
                    vocab_size=vocab_size,
                    params=params,
                    n=n,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training
                )  # [(N,1,D), (N,2,D),(N,4,D),...]
            assert mus[0].shape[1].value == 1
            assert logsigmas[0].shape[1].value == 1
            # Sampling
            if model_mode == 'vae':
                latent_sample, latent_prior_sample = kl_array(
                    mus=mus,
                    logsigmas=logsigmas,
                    params=params,
                    n=n)  # [(N,1,D), (N,2,D),(N,4,D),...]
            elif model_mode == 'aae':
                latent_sample, latent_prior_sample = sample_aae_array(
                    mus=mus,
                    logsigmas=logsigmas)  # [(N,1,D), (N,2,D),(N,4,D),...]
            else:
                raise ValueError("unknown model_mode")
            assert latent_sample[0].shape[1].value == 1
            assert latent_prior_sample[0].shape[1].value == 1
            # Decoder
            with tf.variable_scope('decoder') as decoder_scope:
                logits = decoder_bintree_attention(
                    latent_layers=latent_sample,
                    vocab_size=vocab_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training)  # (L,N,D)
            # Generated data
            with tf.variable_scope(decoder_scope, reuse=True):
                glogits = decoder_bintree_attention(
                    latent_layers=latent_prior_sample,
                    vocab_size=vocab_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training)  # (L,N,D)
        if model_mode == 'aae':
            with tf.variable_scope("discriminator", reuse=False) as discriminator_scope:
                inputs = concat_layers(latent_prior_sample, latent_sample, axis=0)
                dis_out = discriminator_bintree_fn(
                    latent_layers=inputs,
                    params=params,
                    is_training=is_training
                )  # (2n,)
                dis_labels_real = tf.ones(shape=(n,), dtype=tf.float32)
                dis_labels_fake = -tf.ones(shape=(n,), dtype=tf.float32)
                dis_labels = tf.concat([dis_labels_real, dis_labels_fake], axis=0)
                wgan_loss_d = tf.reduce_mean(dis_labels * dis_out)
                wgan_loss_g_raw = -wgan_loss_d
                wgan_scale = get_penalty_scale_logistic(params=params)
                wgan_loss_g = wgan_scale * wgan_loss_g_raw
                tf.summary.scalar("wgan_loss_d", wgan_loss_d)
                tf.summary.scalar("wgan_loss_g_raw", wgan_loss_g_raw)
                tf.summary.scalar("wgan_scale", wgan_scale)
                tf.summary.scalar("wgan_loss_g", wgan_loss_g)
                tf.losses.add_loss(tf.identity(wgan_loss_d, name='wgan_loss_d'))
            with tf.name_scope("autoencoder/"):
                tf.losses.add_loss(tf.identity(wgan_loss_g, name="wgan_loss_g"))

            dis_losses = tf.losses.get_losses(scope=discriminator_scope.name)
            print("Discriminator losses: {}".format(dis_losses))
            dis_losses += tf.losses.get_regularization_losses(scope=discriminator_scope.name)
            dis_total_loss = tf.add_n(dis_losses)
            dis_optimizer = tf.train.AdamOptimizer(params.dis_lr)
            dis_updates = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS, scope=discriminator_scope.name)
            dis_train_op = slim.learning.create_train_op(
                dis_total_loss,
                dis_optimizer,
                clip_gradient_norm=params.clip_gradient_norm,
                variables_to_train=tf.trainable_variables(scope=discriminator_scope.name),
                update_ops=dis_updates,
                global_step=None)
            discriminator_hook = RunTrainOpsHook(
                train_ops=[dis_train_op],
                train_steps=params.discriminator_steps
            )
            training_hooks = [discriminator_hook]
        elif model_mode == 'vae':
            training_hooks = []
        else:
            raise ValueError()

        with tf.control_dependencies([
            tf.assert_equal(
                tf.cast(const_sequence_length, dtype=token_lengths.dtype),
                tf.shape(logits)[0],
                message='tree output shape incorrect')
        ]):
            logits = tf.identity(logits)
        sequence_length_ctc = tf.tile(tf.constant([const_sequence_length], dtype=tf.int32), (n,))

        return ctc_estimator(
            tokens=tokens,
            token_lengths=token_lengths,
            logits=logits,
            glogits=glogits,
            sequence_mask=sequence_mask,
            sequence_length_ctc=sequence_length_ctc,
            vocab=vocab,
            run_config=run_config,
            params=params,
            model_scope=autoencoder_scope.name,
            training_hooks=training_hooks,
            mode=mode)

    return model_fn
