import tensorflow as tf


def infix_indices(depth, stack=[]):
    if depth >= 0:
        left = infix_indices(depth - 1, stack + [0])
        right = infix_indices(depth - 1, stack + [1])
        middle = stack
        return left + [middle] + right
    else:
        return []


def stack_tree(outputs, indices):
    # outputs:[ (N,V), (N,2,V), (N,2,2,V)...]
    # indices:[ paths]

    slices = []
    for idx in indices:
        depth = len(idx)
        output = outputs[depth]
        output_idx = 0
        mult = 1
        for i in reversed(idx):
            output_idx += mult * i
            mult *= 2
        slices.append(output[:, output_idx, :])
    stacked = tf.stack(slices, axis=0)  # (L,N,V)
    return stacked
