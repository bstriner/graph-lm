from .ctc_output import calc_ctc_output
from .utils.bintree_utils import concat_layers, infix_indices, stack_tree
from .utils.bintree_utils_v2 import binary_tree_downward_v2, binary_tree_upward_v2


def decoder_bintree_attention(latent_layers, vocab_size, params, weights_regularizer=None, is_training=True):
    # latent (N, D)
    depth = params.tree_depth
    assert depth >= 0

    hs = latent_layers
    """
    messages_up = binary_tree_upward_v2(
        inputs=hs,
        hidden_dim=params.decoder_dim
    )
    hs = concat_layers(hs, messages_up)
    """
    messages_down = binary_tree_downward_v2(
        inputs=hs,
        hidden_dim=params.decoder_dim
    )
    hs = concat_layers(hs, messages_down)
    indices = infix_indices(depth)
    flat_layers = stack_tree(hs, indices=indices)  # (L,N,V)
    logits = calc_ctc_output(
        flat_layers,
        vocab_size=vocab_size,
        params=params,
        weights_regularizer=weights_regularizer,
        is_training=is_training)
    return logits
