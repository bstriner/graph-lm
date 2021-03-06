import math

import tensorflow as tf
from tensorflow.contrib import slim

from .estimators.ctc_estimator import ctc_estimator
from .estimators.gan_losses import build_gan_losses
from .estimators.gan_train import dis_train_hook
from .estimators.sampling import sampling_flat
from .model_modes import ModelModes
from .networks.decoder_binary_tree import decoder_binary_tree
from .networks.discriminator_output import discriminator_output
from .networks.encoder_flat import encoder_flat
from ..data.word import SENTENCE_LENGTH


def make_model_binary_tree_flat(
        run_config,
        vocabs
):
    vocab = vocabs['text']
    vocab_size = vocab.shape[0]
    print("Vocab size: {}".format(vocab_size))

    def model_fn(features, labels, mode, params):
        model_mode = params.model_mode
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        # Inputs
        tokens = features['text']  # (N, L)
        token_lengths = features[SENTENCE_LENGTH]  # (N,)
        sequence_mask = tf.sequence_mask(maxlen=tf.shape(tokens)[1], lengths=token_lengths)
        n = tf.shape(tokens)[0]
        depth = params.tree_depth
        const_sequence_length = int(math.pow(2, depth + 1) - 1)

        with tf.control_dependencies([
            tf.assert_greater_equal(
                tf.cast(const_sequence_length, dtype=token_lengths.dtype),
                token_lengths,
                message="Tokens longer than tree size"),
            tf.assert_less_equal(
                tokens,
                tf.cast(vocab_size - 1, tokens.dtype),
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

        with tf.variable_scope("autoencoder") as autoencoder_scope:
            embeddings = tf.get_variable(
                dtype=tf.float32,
                name="embeddings",
                shape=[vocab_size, params.embedding_dim],
                initializer=tf.initializers.truncated_normal(
                    stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
            # Encoder
            mu, logsigma = encoder_flat(
                tokens=tokens,
                token_lengths=token_lengths,
                vocab_size=vocab_size,
                params=params,
                n=n,
                weights_regularizer=weights_regularizer,
                is_training=is_training,
                embeddings=embeddings
            )
            # Sampling
            latent_sample, latent_prior_sample = sampling_flat(
                mu=mu,
                logsigma=logsigma,
                params=params,
                n=n)
            # Decoder
            with tf.variable_scope('decoder') as decoder_scope:
                logits = decoder_binary_tree(
                    latent=latent_sample,
                    vocab_size=vocab_size,
                    params=params,
                    embeddings=embeddings,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training)  # (L,N,D)
            if params.model_mode == ModelModes.AE:
                glogits = None
            else:
                with tf.variable_scope(decoder_scope, reuse=True):
                    glogits = decoder_binary_tree(
                        latent=latent_prior_sample,
                        vocab_size=vocab_size,
                        params=params,
                        embeddings=embeddings,
                        weights_regularizer=weights_regularizer,
                        is_training=is_training)  # (L,N,D)
        if params.model_mode == ModelModes.AAE_RE or params.model_mode == ModelModes.AAE_STOCH:
            with tf.variable_scope("discriminator", reuse=False) as discriminator_scope:
                dis_inputs = tf.concat([latent_prior_sample, latent_sample], axis=0)
                dis_out = discriminator_output(
                    x=dis_inputs,
                    params=params,
                    is_training=is_training
                )  # (2n,)
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
        elif model_mode == ModelModes.VAE:
            training_hooks = []
        elif model_mode == ModelModes.AE:
            training_hooks = []
        else:
            raise ValueError()
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
