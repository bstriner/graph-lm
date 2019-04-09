import math

import tensorflow as tf
from tensorflow.contrib import slim

from .estimators.vae_ctc_estimator import ctc_estimator
from .networks.decoder_bintree_attention import decoder_bintree_attention
from .networks.encoder_bintree_attention import encoder_bintree_attn
from .networks.encoder_bintree_recurrent_attention import encoder_bintree_recurrent_attn_vae
from ..data.word import SENTENCE_LENGTH, TEXT
from ..kl import kl_array


def make_model_vae_binary_tree_attn(
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
        latent_sample, latent_prior_sample = kl_array(
            mus=mus,
            logsigmas=logsigmas,
            params=params,
            n=n)  # [(N,1,D), (N,2,D),(N,4,D),...]
        assert latent_sample[0].shape[1].value == 1
        assert latent_prior_sample[0].shape[1].value == 1
        # Decoder
        with tf.variable_scope('vae_decoder') as decoder_scope:
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

        # print("Lengths: {} vs {}".format(len(indices), math.pow(2, depth + 1) - 1))
        # assert len(indices) == int(math.pow(2, depth + 1) - 1)
        # assert max(len(i) for i in indices) == depth
        # assert len(tree_layers) == depth + 1
        with tf.control_dependencies([
            tf.assert_equal(
                tf.cast(const_sequence_length, dtype=token_lengths.dtype),
                tf.shape(logits)[0],
                message='tree output shape incorrect')
        ]):
            logits = tf.identity(logits)
        sequence_length_ctc = tf.tile(tf.constant([int(math.pow(2, depth + 1) - 1)], dtype=tf.int32), (n,))

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
            mode=mode)

    return model_fn
