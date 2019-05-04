import tensorflow as tf

from .aae import sample_aae
from .kl import kl
from ..model_modes import ModelModes


def sampling_flat(params, mu, logsigma, n):
    if params.model_mode == ModelModes.AAE_STOCH:
        return mu, tf.random_normal(shape=tf.shape(mu))
    elif params.model_mode == ModelModes.VAE:
        return kl(
            mu=mu,
            logsigma=logsigma,
            params=params,
            n=n
        )
    elif params.model_mode == ModelModes.AAE_RE:
        return sample_aae(mu=mu, logsigma=logsigma)
    elif params.model_mode == ModelModes.AE:
        return mu, tf.random_normal(shape=tf.shape(mu))
    else:
        raise ValueError()
