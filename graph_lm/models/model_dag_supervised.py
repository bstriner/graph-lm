import tensorflow as tf
from tensorflow.contrib import slim

from graph_lm.models.estimators.kl import kl
from graph_lm.models.networks.utils.dag_utils import build_dag
from .estimators.aae import sample_aae
from .estimators.crossentropy_estimator import crossentropy_estimator
from .estimators.gan_losses import build_gan_losses
from .estimators.gan_train import dis_train_hook
from .networks.decoder_dag_supervised import decoder_dag_supervised
from .networks.discriminator_dag_supervised import discriminator_dag_supervised
from .networks.encoder_dag import encoder_dag
from ..data.word import SENTENCE_LENGTH, TEXT


def make_model_dag_supervised(
        run_config,
        vocabs,
        model_mode='vae'
):
    vocab = vocabs[TEXT]
    vocab_size = vocab.shape[0]
    print("Vocab size: {}".format(vocab_size))

    def model_fn(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        print("Features: {}".format(features))
        # Inputs
        sequence_length = features[SENTENCE_LENGTH]
        text = features[TEXT]
        # tags = features['tags']
        heads = features['head']  # (N,L)
        sequence_mask = tf.sequence_mask(maxlen=tf.shape(text)[1], lengths=sequence_length)  # (N,L)
        n = tf.shape(text)[0]
        L = tf.shape(text)[1]

        dag = build_dag(heads=heads)  # (n, dep, head)
        dag_bw = tf.transpose(dag, (0, 2, 1))  # (n, head, dep)

        Lf = tf.cast(L, tf.float32)
        h_linspace = tf.cast(tf.range(tf.cast(L, tf.int32)), tf.float32) #[0 ... L-1]
        h_linspace_tiled = tf.tile(tf.expand_dims(h_linspace, 0), [n, 1])  # (N, L)
        h_linspace_scaled = h_linspace_tiled * (Lf / (Lf-1)) / tf.expand_dims(tf.cast(sequence_length, tf.float32), -1)
        #h_head_diff = tf.cast(heads, tf.float32) - tf.cast(tf.range(tf.cast(L, tf.int32)), tf.float32) -1.
        h_headcount = tf.reduce_sum(dag, axis=1) # (N, head)
        h_childcount = tf.reduce_sum(dag, axis=2) # (N, child)
        h_headpos = tf.reduce_sum(dag * h_linspace, -1)
        h_diff = h_headpos - h_linspace
        dag_feats = tf.stack([
            h_linspace_tiled, h_linspace_scaled, h_headcount,
            h_childcount, h_headpos, h_diff
        ], axis=-1)

        with tf.control_dependencies([
            tf.assert_greater(tf.constant(vocab_size, dtype=text.dtype), text, message="Tokens larger than vocab"),
            tf.assert_greater_equal(text, tf.constant(0, dtype=text.dtype), message="Tokens less than 0")
        ]):
            text = tf.identity(text)

        if params.l2 > 0:
            weights_regularizer = slim.l2_regularizer(params.l2)
        else:
            weights_regularizer = None

        with tf.variable_scope('autoencoder') as autoencoder_scope:

            with tf.variable_scope('encoder'):
                # Encoder
                mu, logsigma = encoder_dag(
                    dag=dag,
                    dag_bw=dag_bw,
                    dag_feats=dag_feats,
                    text=text,
                    vocab_size=vocab_size,
                    params=params,
                    weights_regularizer=weights_regularizer
                )  # (N,L,D)

            # Sample
            idx = tf.where(sequence_mask)
            with tf.name_scope("sampling"):
                selected_mu = tf.gather_nd(params=mu, indices=idx)
                selected_logsigma = tf.gather_nd(params=logsigma, indices=idx)
                if model_mode == 'aae':
                    latent_sample_values, latent_prior_sample_values = sample_aae(
                        mu=selected_mu,
                        logsigma=selected_logsigma)
                elif model_mode == 'vae':
                    latent_sample_values, latent_prior_sample_values = kl(
                        mu=selected_mu,
                        logsigma=selected_logsigma,
                        params=params,
                        n=n)
                else:
                    raise ValueError()
                latent_sample = tf.scatter_nd(
                    updates=latent_sample_values,
                    indices=idx,
                    shape=(n, L, latent_sample_values.shape[-1].value)
                )  # (N,L,D)
                latent_prior_sample = tf.scatter_nd(
                    updates=latent_prior_sample_values,
                    indices=idx,
                    shape=(n, L, latent_prior_sample_values.shape[-1].value)
                )  # (N,L,D)

            # Decoder
            with tf.variable_scope('vae_decoder', reuse=False) as decoder_scope:
                logits = decoder_dag_supervised(
                    latent=latent_sample,
                    vocab_size=vocab_size,
                    dag=dag,
                    dag_bw=dag_bw,
                    dag_feats=dag_feats,
                    sequence_length=sequence_length,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training
                )
            with tf.variable_scope(decoder_scope, reuse=True):
                glogits = decoder_dag_supervised(
                    latent=latent_prior_sample,
                    vocab_size=vocab_size,
                    dag=dag,
                    dag_bw=dag_bw,
                    dag_feats=dag_feats,
                    sequence_length=sequence_length,
                    params=params,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training
                )
        if model_mode == 'aae':
            with tf.variable_scope("discriminator", reuse=False) as discriminator_scope:
                dis_fake = discriminator_dag_supervised(
                    latent=latent_sample,
                    dag=dag,
                    dag_bw=dag_bw,
                    dag_feats=dag_feats,
                    sequence_length=sequence_length,
                    params=params,
                    idx=idx,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training)
            with tf.variable_scope(discriminator_scope, reuse=True):
                dis_real = discriminator_dag_supervised(
                    latent=latent_prior_sample,
                    dag=dag,
                    dag_bw=dag_bw,
                    dag_feats=dag_feats,
                    sequence_length=sequence_length,
                    params=params,
                    idx=idx,
                    weights_regularizer=weights_regularizer,
                    is_training=is_training)
                dis_out = tf.concat([dis_real, dis_fake], axis=-1)
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
        elif model_mode == 'vae':
            with tf.name_scope(autoencoder_scope.name + "/"):
                pass
        else:
            raise ValueError()

        return crossentropy_estimator(
            tokens=text,
            token_lengths=sequence_length,
            logits=logits,
            glogits=glogits,
            idx=idx,
            vocab=vocab,
            run_config=run_config,
            params=params,
            model_scope=autoencoder_scope.name,
            training_hooks=training_hooks,
            mode=mode)

    return model_fn
