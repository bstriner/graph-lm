import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def sample_aae(mu, logsigma):
    one_fix = False
    if mu.shape[1].value == 1:
        mu = tf.squeeze(mu, 1)
        logsigma = tf.squeeze(logsigma, 1)
        one_fix = True
        print("Fixing ones issue")
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
    # print("latent_sample: {}".format(latent_sample))
    if one_fix:
        latent_sample = tf.expand_dims(latent_sample, 1)
        latent_prior_sample = tf.expand_dims(latent_prior_sample, 1)
    return latent_sample, latent_prior_sample


def sample_aae_array(mus, logsigmas):
    rets = [sample_aae(
        mu=mu,
        logsigma=logsigma
    ) for mu, logsigma in zip(mus, logsigmas)]
    rets = [list(l) for l in zip(*rets)]
    return rets


