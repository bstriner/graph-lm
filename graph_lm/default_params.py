import json
import os

import six
import tensorflow as tf
from tensorflow.contrib.training import HParams


def load_hparams(model_dir):
    hparams = default_params()
    hparams_path = os.path.join(model_dir, 'configuration-hparams.json')
    assert os.path.exists(hparams_path)

    with open(hparams_path) as f:
        return hparams.parse_json(f.read())


def get_hparams(model_dir, validate=True):
    config_file = tf.flags.FLAGS.config
    hparams = default_params()
    hparams_path = os.path.join(model_dir, 'configuration-hparams.json')
    with open(config_file) as f:
        hparams.parse_json(f.read())

    if os.path.exists(hparams_path):
        if validate:
            with open(hparams_path) as f:
                hparam_dict = json.load(f)
            for k, v in six.iteritems(hparam_dict):
                oldval = getattr(hparams, k)
                assert oldval == v, "Incompatible key {}: save {}-> config {}".format(k, oldval, v)
    else:
        with open(hparams_path, 'w') as f:
            json.dump(hparams.values(), f)
    return hparams


def default_params():
    return HParams(
        model='ctasdsac',
        model_mode="vae",
        gan_loss='v4',

        encoder_dim=256,
        encoder_layers=3,
        decoder_dim=256,
        decoder_layers=2,
        discriminator_dim=256,
        discriminator_layers=2,
        latent_dim=256,
        attention_dim=128,
        noise_dim=256,

        batch_norm=False,
        infix_tree=False,
        lstm_output=False,
        lstm_output_discriminator=False,
        bias_smoothing=0.05,

        tree_depth=8,
        flat_length=300,
        discriminator_steps=5,
        lr=3e-5,
        dis_lr=3e-5,
        l2=1e-7,
        clip_gradient_norm=1.,
        series_depth=8,
        message_depth=8,
        dag_penalty_weight=1e-3,

        kl_anneal_start=5000,
        kl_anneal_end=200000,
        kl_anneal_min=1e-4,
        kl_anneal_max=1.,
        penalty_anneal_start=5000,
        penalty_anneal_end=200000,
        penalty_anneal_min=1e-4,
        penalty_anneal_max=1e-4,

        model_version='v1',
        attn_mode='softmax',
        subsample=3,
        depth=6,
        listener_dim=320,
        dropout=0.,
        vae_depth=3,
        vae_dropout=0.,
        vae_dim=320,

        kl_min=1e-2,
        kernel_size=7
    )


"""
        encoder_dim=256,
        # query_dim=128,
        # value_dim=128,
        # decoder_dim=256,
        latent_dim=128,
        attention_dim=128,


        discriminator_dim=128,
        dis_lr=3e-4,
        gen_lr=3e-4,
        dis_steps=5
        """
