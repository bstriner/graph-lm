import tensorflow as tf
import tensorflow_probability as tfp

from ...anneal import get_kl_scale_logistic

tfd = tfp.distributions


def calc_kl_loss_raw(mu, logsigma, params, n):
    one_fix=False
    if mu.shape[1].value==1:
        mu = tf.squeeze(mu, 1)
        logsigma = tf.squeeze(logsigma, 1)
        one_fix=True
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
    #print("latent_sample: {}".format(latent_sample))
    if one_fix:
        latent_sample=tf.expand_dims(latent_sample, 1)
        latent_prior_sample=tf.expand_dims(latent_prior_sample, 1)
    # scale_identity_multiplier=1.0)
    #assert logsigma.shape[1].value == mu.shape[1].value
    #assert latent_sample.shape[1].value == mu.shape[1].value

    kl_n = tfd.kl_divergence(latent_dist, latent_prior)
    kl_n = tf.maximum(kl_n, params.kl_min)  # (L, N)
    #print("kl_n shape: {}".format(kl_n))
    kl_loss_raw = tf.reduce_sum(kl_n) / tf.cast(n, tf.float32)
    return latent_sample, latent_prior_sample, kl_loss_raw


def kl(mu, logsigma, params, n):
    latent_sample, latent_prior_sample, kl_loss_raw = calc_kl_loss_raw(
        mu=mu,
        logsigma=logsigma,
        params=params,
        n=n
    )
    kl_scale = get_kl_scale_logistic(params)
    kl_loss = kl_scale * kl_loss_raw
    tf.summary.scalar("kl_raw", kl_loss_raw)
    tf.summary.scalar("kl_scale", kl_scale)
    tf.summary.scalar("kl_weighted", kl_loss)
    if params.kl_anneal_max > 0:
        tf.losses.add_loss(kl_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    return latent_sample, latent_prior_sample


def kl_array(mus, logsigmas, params, n):
    rets = [calc_kl_loss_raw(
        mu=mu,
        logsigma=logsigma,
        params=params,
        n=n
    ) for mu, logsigma in zip(mus, logsigmas)]
    latent_samples, latent_prior_samples, kl_loss_raws = [list(l) for l in zip(*rets)]
    kl_loss_raw = tf.add_n(kl_loss_raws)
    kl_scale = get_kl_scale_logistic(params)
    kl_loss = kl_scale * kl_loss_raw
    tf.summary.scalar("kl_raw", kl_loss_raw)
    tf.summary.scalar("kl_scale", kl_scale)
    tf.summary.scalar("kl_weighted", kl_loss)
    if params.kl_anneal_max > 0:
        tf.losses.add_loss(kl_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    return latent_samples, latent_prior_samples
