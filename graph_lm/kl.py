import tensorflow as tf
import tensorflow_probability as tfp

from .anneal import get_scale

tfd = tfp.distributions


def kl(mu, logsigma, params, n):
    latent_dist = tfd.MultivariateNormalDiag(
        loc=mu,
        scale_diag=tf.nn.softplus(logsigma),
        name="latent_dist")
    latent_sample = latent_dist.sample()
    assert latent_sample.shape.ndims == mu.shape.ndims
    latent_prior = tfd.MultivariateNormalDiag(
        loc=tf.zeros_like(mu),
        scale_diag=tf.ones_like(logsigma),
        name="latent_prior")
    latent_prior_sample = latent_prior.sample()
    print("latent_sample: {}".format(latent_sample))
    # scale_identity_multiplier=1.0)

    kl_n = tfd.kl_divergence(latent_dist, latent_prior)
    kl_n = tf.maximum(kl_n, params.kl_min) # (L, N)
    print("kl_n shape: {}".format(kl_n))
    kl_loss_raw = tf.reduce_sum(kl_n) / tf.cast(n, tf.float32)
    kl_scale = get_scale(params)
    kl_loss = kl_loss_raw * kl_scale

    tf.summary.scalar("kl_loss_raw", kl_loss_raw)
    tf.summary.scalar("kl_scale", kl_scale)
    tf.summary.scalar("kl_loss_scaled", kl_loss)
    tf.losses.add_loss(kl_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    return latent_sample, latent_prior_sample
