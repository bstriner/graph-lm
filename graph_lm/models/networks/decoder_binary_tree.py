from tensorflow.contrib import slim

from .ctc_output import calc_ctc_output
from .utils.bintree_utils import binary_tree_resnet, infix_indices, stack_tree_v2


def decoder_binary_tree(latent, vocab_size, params, weights_regularizer=None, is_training=True, reuse=None):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0
    h = slim.fully_connected(
        latent,
        num_outputs=params.decoder_dim,
        scope='projection',
        activation_fn=None,
        weights_regularizer=weights_regularizer
    )
    tree_layers = binary_tree_resnet(
        x0=h,
        depth=params.tree_depth,
        hidden_dim=params.decoder_dim
    )
    flat_layers = stack_tree_v2(tree_layers)  # (L,N,V)
    logits = calc_ctc_output(
        flat_layers,
        vocab_size=vocab_size,
        params=params,
        weights_regularizer=weights_regularizer,
        is_training=is_training)
    return logits
