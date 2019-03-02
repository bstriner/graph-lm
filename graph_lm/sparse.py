import tensorflow as tf


def sparse_to_counts(sparse: tf.SparseTensor, axis=-1):
    # assert sparse.dense_shape.shape[0].value == 2
    ind_sparse = tf.SparseTensor(
        indices=sparse.indices,
        dense_shape=sparse.dense_shape,
        values=tf.ones(shape=tf.shape(sparse.values), dtype=tf.int32)
    )
    counts = tf.sparse_reduce_sum(
        sp_input=ind_sparse,
        axis=axis
    )
    return counts


def sparsify(x, mask):
    selected_idx = tf.where(mask)
    selected_values = tf.gather_nd(
        params=x,
        indices=selected_idx
    )
    selected_idx = tf.cast(selected_idx, dtype=tf.int64)
    dense_shape = tf.shape(x, out_type=tf.int64)
    sparse = tf.SparseTensor(
        indices=selected_idx,
        values=selected_values,
        dense_shape=dense_shape
    )
    return sparse


def sparse_mask_indicator(x):
    out = tf.scatter_nd(
        updates=tf.ones_like(x.values),
        indices=x.indices,
        shape=x.dense_shape
    )
    return out


def sparse_tensor_to_dense_scatter(sp_input: tf.SparseTensor):
    return tf.scatter_nd(
        indices=sp_input.indices,
        updates=sp_input.values,
        shape=sp_input.dense_shape,
    )


if __name__ == '__main__':
    with tf.Session() as sess:
        x = tf.constant([
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
        ])
        mask = tf.constant([
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1]
        ])
        mask = tf.cast(mask, dtype=tf.bool)

        s = sparsify(x, mask)
        d = tf.sparse_tensor_to_dense(s)
        ss = sess.run(d)
        print(ss)

        sm = sparse_mask_indicator(s)
        print(sess.run(sm))
