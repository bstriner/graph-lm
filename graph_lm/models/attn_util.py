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

    b_lengths_ta = tf.TensorArray(size=n, dtype=b_lengths.dtype)
    b_lengths_ta = b_lengths_ta.unstack(b_lengths)
    energy_ta = tf.TensorArray(size=n, dtype=energy.dtype)
    energy_ta = energy_ta.unstack(energy)

    def body(i, arr: tf.TensorArray):
        energy_t = energy_ta.read(i)
        b_lengths_t = b_lengths_ta.read(i)
        attn = tf.nn.softmax(energy_t[:, :b_lengths_t], axis=-1)
        pattn = tf.pad(attn, [[0, 0], [0, bl - b_lengths_t]])
        return i + tf.constant(1, dtype=tf.int32), arr.write(i, pattn)

    loop_vars_in = tf.constant(0, dtype=tf.int32), tf.TensorArray(size=n, dtype=a.dtype)
    i_out, arr_out = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars_in
    )
    return arr_out.stack()  # (n, al, bl)
