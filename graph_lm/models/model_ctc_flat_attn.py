import os

import tensorflow as tf
from tensorflow.contrib import slim

from .networks.decoder_flat_attn import vae_flat_decoder_attn
from .networks.encoder_flat_attn import encoder_flat_attn
from ..callbacks.ctc_callback import CTCHook
from graph_lm.models.estimators.kl import kl
from ..sparse import sparsify

from .estimators.ctc_estimator import ctc_estimator
from .estimators.gan_losses import build_gan_losses
from .estimators.gan_train import dis_train_hook
from .estimators.sampling import sampling_flat
from .model_modes import ModelModes
from .networks.decoder_flat import decoder_flat
from .networks.discriminator_output import discriminator_output
from .networks.encoder_flat import encoder_flat
from ..data.word import SENTENCE_LENGTH, TEXT

def make_model_ctc_flat_attn(
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


        with tf.variable_scope('autoencoder') as autoencoder_scope:
            # Encoder
            mu, logsigma = encoder_flat_attn(
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
            latent_sample, latent_prior_sample = sampling_flat(
                mu=mu,
                logsigma=logsigma,
                params=params,
                n=n)

            # Decoder
            with tf.variable_scope('decoder', reuse=False) as decoder_scope:
                logits = vae_flat_decoder_attn(
                    latent=latent_sample,
                    vocab_size=vocab_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    n=n,
                    is_training=is_training
                )
            # Generated data
            with tf.variable_scope(decoder_scope, reuse=True):
                glogits = vae_flat_decoder_attn(
                    latent=latent_prior_sample,
                    vocab_size=vocab_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    n=n,
                    is_training=is_training
                )

        if params.model_mode == ModelModes.AAE_RE or params.model_mode == ModelModes.AAE_STOCH:
            with tf.variable_scope('discriminator') as discriminator_scope:
                dis_inputs = tf.concat([latent_prior_sample, latent_sample], axis=0)
                dis_inputs = tf.reshape(dis_inputs, (n, dis_inputs.shape[1].value*dis_inputs.shape[2].value))
                dis_out = discriminator_output(
                    x=dis_inputs, params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training)
                dis_out = tf.squeeze(dis_out, -1)
                print("Dis: {} -> {}".format(dis_inputs, dis_out))
            build_gan_losses(
                params=params,
                autoencoder_scope=autoencoder_scope.name,
                discriminator_scope=discriminator_scope.name,
                dis_out=dis_out,
                n=n
            )
            discriminator_hook = dis_train_hook(
                discriminator_scope=discriminator_scope.name,
                params=params
            )
            training_hooks = [discriminator_hook]
        elif params.model_mode == ModelModes.VAE:
            training_hooks = []
        elif params.model_mode == ModelModes.AE:
            training_hooks = []
        else:
            raise ValueError()

        sequence_length_ctc = tf.tile([params.flat_length], (n,))  # tf.shape(logits)[0:1], (n,))

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
