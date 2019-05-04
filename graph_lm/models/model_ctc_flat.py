import tensorflow as tf
from tensorflow.contrib import slim

from .estimators.ctc_estimator import ctc_estimator
from .estimators.gan_losses import build_gan_losses
from .estimators.gan_train import dis_train_hook
from .estimators.sampling import sampling_flat
from .model_modes import ModelModes
from .networks.decoder_flat import decoder_flat
from .networks.discriminator_output import discriminator_output
from .networks.encoder_flat import encoder_flat
from ..data.word import SENTENCE_LENGTH, TEXT


def make_model_ctc_flat(
        run_config,
        vocabs
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
        length = params.flat_length

        with tf.control_dependencies([
            tf.assert_greater_equal(length, token_lengths, message="Tokens longer than flat_length"),
            tf.assert_less_equal(tokens, tf.cast(vocab_size - 1, dtype=tokens.dtype),
                                 message="Tokens larger than vocab"),
            tf.assert_greater_equal(tokens, tf.cast(0, dtype=tokens.dtype), message="Tokens less than 0")
        ]):
            tokens = tf.identity(tokens)

        if params.l2 > 0:
            weights_regularizer = slim.l2_regularizer(params.l2)
        else:
            weights_regularizer = None

        with tf.variable_scope('autoencoder') as autoencoder_scope:
            # Encoder
            with tf.variable_scope('encoder'):
                mu, logsigma = encoder_flat(
                    tokens=tokens,
                    token_lengths=token_lengths,
                    vocab_size=vocab_size,
                    params=params,
                    n=n,
                    weights_regularizer=weights_regularizer
                )
            # Sampling
            latent_sample, latent_prior_sample = sampling_flat(
                mu=mu,
                logsigma=logsigma,
                params=params,
                n=n)

            # Decoder
            with tf.variable_scope('decoder', reuse=False) as decoder_scope:
                logits = decoder_flat(
                    latent=latent_sample,
                    vocab_size=vocab_size,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    n=n
                )
            if params.model_mode == ModelModes.AE:
                glogits = None
            else:
                with tf.variable_scope(decoder_scope, reuse=True):
                    glogits = decoder_flat(
                        latent=latent_prior_sample,
                        vocab_size=vocab_size,
                        params=params,
                        weights_regularizer=weights_regularizer,
                        n=n
                    )

        if params.model_mode == ModelModes.AAE_RE or params.model_mode == ModelModes.AAE_STOCH:
            with tf.variable_scope('discriminator') as discriminator_scope:
                dis_inputs = tf.concat([latent_sample, latent_prior_sample], axis=0)
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
        sequence_length_ctc = tf.tile([length], (n,))

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
