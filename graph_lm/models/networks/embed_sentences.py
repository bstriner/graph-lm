import tensorflow as tf


def embed_sentences(tokens, token_lengths, vocab_size, params):
    n = tf.shape(tokens)[1]
    L = tf.shape(tokens)[0]
    embeddings = tf.get_variable(
        dtype=tf.float32,
        name="embeddings",
        shape=[vocab_size, params.encoder_dim],
        initializer=tf.initializers.truncated_normal(
            stddev=1. / tf.sqrt(tf.constant(params.encoder_dim, dtype=tf.float32))))
    h = tf.nn.embedding_lookup(embeddings, tokens)  # (L, N, D)
    ls = tf.linspace(
        start=tf.constant(0, dtype=tf.float32),
        stop=tf.constant(1, dtype=tf.float32),
        num=L)  # (L,)
    ls = tf.tile(tf.expand_dims(ls, 1), [1, n])  # (L,N)
    ls = ls * tf.cast(L, dtype=tf.float32) / tf.cast(tf.expand_dims(token_lengths, 0), dtype=tf.float32)
    ls = tf.expand_dims(ls, 2)  # ( L,N,1)
    h = tf.concat([h, ls], axis=-1)
    return h
