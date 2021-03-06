import tensorflow as tf

from .ctc_output import calc_ctc_output
from .utils.bintree_utils import concat_layers, stack_tree_v2
from .utils.bintree_utils_v2 import binary_tree_downward_v2
from .utils.rnn_util import lstm
from tensorflow.contrib import slim

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
    if params.infix_tree:
        flat_layers = stack_tree_v2(hs)  # (L,N,V)
    else:
        flat_layers = tf.transpose(hs[-1], (1, 0, 2))
    if params.lstm_output:
        h, _ = lstm(
            x=flat_layers,
            num_units=params.decoder_dim,
            bidirectional=True,
            num_layers=params.decoder_layers
        )
        logits = slim.fully_connected(
            inputs=h,
            num_outputs=vocab_size + 1,
            activation_fn=None,
            scope='output_mlp_3',
            weights_regularizer=weights_regularizer
            #biases_initializer=tf.initializers.constant(get_bias_ctc(
            #    average_output_length=math.pow(2, params.tree_depth + 1) - 1,
            #    smoothing=0.05
            #))
        )
    else:
        logits = calc_ctc_output(
            flat_layers,
            vocab_size=vocab_size,
            params=params,
            weights_regularizer=weights_regularizer,
            is_training=is_training)
    return logits
