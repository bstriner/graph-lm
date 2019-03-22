import tensorflow as tf


def calc_attn(a, b, b_lengths):
    # a: (la,n, d)
    # b: (lb,n, d)
    # b_lengths: (n,)

    bl = tf.shape(b)[0]
    n = tf.shape(b)[1]
    energy = tf.matmul(
        tf.transpose(a, (1, 0, 2)),  # (n, al, d)
        tf.transpose(b, (1, 2, 0))  # (n, d, bl)
    )  # (n, al, bl)

    def cond(i, arr):
        return i < n

    def body(i, arr: tf.TensorArray):
        blen = b_lengths[i]
        attn = tf.nn.softmax(energy[i, :, :blen], axis=-1)
        pattn = tf.pad(attn, [[0, 0], [0, bl - blen]])
        return i + 1, arr.write(i, pattn)

    loop_vars_in = tf.constant(0, dtype=tf.int32), tf.TensorArray(size=n, dtype=a.dtype)
    i_out, arr_out = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars_in
    )
    return arr_out.stack() # (n, al, bl)
