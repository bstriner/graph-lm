from .utils.bintree_utils import concat_layers, infix_indices, stack_tree
import tensorflow as tf

from .discriminator_output import discriminator_output
from .utils.bintree_utils import concat_layers, infix_indices, stack_tree
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
    indices = infix_indices(depth)
    flat_layers = stack_tree(hs, indices=indices)  # (L,N,D)
    logits = discriminator_output(
        flat_layers,
        params=params,
        weights_regularizer=weights_regularizer,
        is_training=is_training)  # (L,N,1)
    logits = tf.reduce_mean(logits, axis=0)  # (N,1)
    logits = tf.squeeze(logits, axis=1)  # (N,)
    return logits
