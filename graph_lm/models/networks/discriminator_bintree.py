import tensorflow as tf

from .discriminator_output import discriminator_output
from .utils.bintree_utils import concat_layers
from .utils.bintree_utils_v2 import binary_tree_downward_v2, binary_tree_upward_v2
from ...sn import sn_fully_connected


def discriminator_bintree_fn(
        latent_layers,
        params,
        weights_regularizer=None,
        is_training=True):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0
    assert len(latent_layers) == depth + 1

    hs = latent_layers
    messages_up = binary_tree_upward_v2(
        inputs=hs,
        hidden_dim=params.decoder_dim,
        fc_fn=sn_fully_connected
    )
    hs = concat_layers(hs, messages_up)
    messages_down = binary_tree_downward_v2(
        inputs=hs,
        hidden_dim=params.decoder_dim,
        fc_fn=sn_fully_connected
    )
    hs = concat_layers(hs, messages_down)
    h = tf.concat(hs, axis=1)  # (N, L, D)
    logits = discriminator_output(
        h,
        params=params,
        weights_regularizer=weights_regularizer,
        is_training=is_training)  # (N,L,1)
    logits = tf.reduce_sum(logits, axis=1)  # (N,1)
    logits = tf.squeeze(logits, axis=1)  # (N,)
    return logits
