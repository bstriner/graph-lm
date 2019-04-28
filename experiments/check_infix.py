import timeit

import tensorflow as tf

from graph_lm.models.networks.utils.bintree_utils import flat_infix_indices, infix_indices, stack_tree
import numpy as np

def check(depth=8, n=1, dim=32, number=100):
    idx = infix_indices(depth=depth)
    print(idx)
    ids = flat_infix_indices(depth=depth)
    print(ids)

    xs = [
        tf.random_normal(shape=(n, 2 ** i, dim))
        for i in range(depth + 1)
    ]

    # v1
    v1 = stack_tree(xs, indices=idx)  # (L,N,V)

    # v2
    h = tf.concat(xs, axis=1)
    h = tf.transpose(h, (1, 0, 2))
    v2 = tf.gather(params=h, indices=ids, axis=0)


    with tf.train.MonitoredSession() as sess:
        x1, x2 = sess.run([v1,v2])
        assert np.allclose(x1, x2)

        _ = sess.run(v1)

        v1t = timeit.timeit(stmt=lambda: sess.run(v1), number=number) / number

        _ = sess.run(v2)

        v2t = timeit.timeit(stmt=lambda: sess.run(v2), number=number) / number

        print("V1: {}".format(v1t))
        print("V2: {}".format(v2t))



def main():
    check(depth=10, n=8)


if __name__ == '__main__':
    main()
