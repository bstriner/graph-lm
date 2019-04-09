import tensorflow as tf

from .ctc_output import calc_ctc_output
from .utils.bintree_utils import binary_tree_up, binary_tree_down, infix_indices, stack_tree


def decoder_bintree_attention(latent_layers, vocab_size, params, weights_regularizer=None, is_training=True):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0

    hs = latent_layers
    messages_up = binary_tree_up(
        inputs=hs,
        hidden_dim=params.decoder_dim
    )
    hs = [
        tf.concat(cols, axis=-1)
        for cols in zip(hs, messages_up)
    ]
    messages_down = binary_tree_down(
        x0=tf.squeeze(hs[0], axis=1),
        inputs=hs,
        hidden_dim=params.decoder_dim,
        depth=depth
    )
    hs = [
        tf.concat(cols, axis=-1)
        for cols in zip(hs, messages_down)
    ]
    indices = infix_indices(depth)
    flat_layers = stack_tree(hs, indices=indices)  # (L,N,V)
    logits = calc_ctc_output(
        flat_layers,
        vocab_size=vocab_size,
        params=params,
        weights_regularizer=weights_regularizer,
        is_training=is_training)
    return logits
