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

    # an = tf.norm(ord=2, axis=-1, tensor=a)  # (la, n)
    # bn = tf.norm(ord=2, axis=-1, tensor=b)  # (lb, n)
    # energy = energy / tf.expand_dims(tf.transpose(an, (1, 0)), axis=2)
    # energy = energy / tf.expand_dims(tf.transpose(bn, (1, 0)), axis=1)

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


def calc_attn_v2(a, b, b_lengths, a_transpose=True, b_transpose=True):
    # a: (la,n, d)
    # b: (lb,n, d)
    # b_lengths: (n,)
    if a_transpose:
        n = tf.shape(a)[1]
        a_ta = tf.TensorArray(size=n, dtype=a.dtype).unstack(tf.transpose(a, (1, 0, 2)))  # (n, la, d)
    else:
        n = tf.shape(a)[0]
        a_ta = tf.TensorArray(size=n, dtype=a.dtype).unstack(a)  # (n, la, d)
    if b_transpose:
        bl = tf.shape(b)[0]
        b_ta = tf.TensorArray(size=n, dtype=b.dtype).unstack(tf.transpose(b, (1, 0, 2)))  # (n, lb, d)
    else:
        bl = tf.shape(b)[1]
        b_ta = tf.TensorArray(size=n, dtype=b.dtype).unstack(b)  # (n, lb, d)
    b_lengths_ta = tf.TensorArray(size=n, dtype=b_lengths.dtype).unstack(b_lengths)  # (n,)

    def cond(i, arr):
        return i < n

    def body(i, arr: tf.TensorArray):
        a_t = a_ta.read(i)
        #a_n = tf.norm(a_t, axis=-1, ord=2)
        b_t = b_ta.read(i)
        b_lengths_t = b_lengths_ta.read(i)
        b_part = b_t[:b_lengths_t, :]  # (bl, d)
        b_n = tf.norm(b_part, axis=-1, ord=2)
        energy = tf.tensordot(a_t, b_part, axes=[(1,), (1,)])  # (al, bl)
        energy = energy / tf.expand_dims(b_n, 0)
        attn = tf.nn.softmax(energy, axis=-1)
        pattn = tf.pad(attn, [[0, 0], [0, bl - b_lengths_t]])
        return i + tf.constant(1, dtype=tf.int32), arr.write(i, pattn)

    loop_vars_in = tf.constant(0, dtype=tf.int32), tf.TensorArray(size=n, dtype=a.dtype)
    i_out, arr_out = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=loop_vars_in
    )
    return arr_out.stack()  # (n, al, bl)
