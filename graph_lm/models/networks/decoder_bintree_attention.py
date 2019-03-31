from tensorflow.contrib import slim

from .tree_utils import infix_indices, stack_tree
from .bintree_utils import binary_tree_resnet

from .ctc_output import calc_output
import tensorflow as tf


def decoder_bintree_attention(latent_layers, vocab_size, params, weights_regularizer=None, is_training=True):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0
    x0 = tf.squeeze(latent_layers[0], axis=1)
    tree_layers = binary_tree_resnet(
        x0=x0,
        depth=params.tree_depth,
        hidden_dim=params.decoder_dim,
        inputs=latent_layers
    )
    indices = infix_indices(depth)
    flat_layers = stack_tree(tree_layers, indices=indices)  # (L,N,V)
    logits = calc_output(
        flat_layers,
        vocab_size=vocab_size,
        params=params,
        weights_regularizer=weights_regularizer,
        is_training=is_training)
    return tree_layers, logits
