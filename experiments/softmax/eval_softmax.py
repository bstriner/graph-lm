import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

V = 1000
N = 1000
D = 6
I = 50000


def activation_softmax(h):
    return h - tf.expand_dims(tf.nn.log_softmax(h, axis=-1), axis=-1)


def activation_sigsoftmax(h):
    pass


def model_fn(softmax_fn=activation_softmax):
    """
    x = tf.random_uniform(
        minval=0,
        maxval=V,
        shape=(N,),
        dtype=tf.int64
    )
    """
    x = tf.range(
        start=0,
        limit=V,
        dtype=tf.int64
    )
    x_one_hot = tf.one_hot(
        indices=x,
        depth=V
    )
    embeddings = tf.get_variable(
        dtype=tf.float32,
        name="embeddings",
        shape=[V, D])
    embedded = tf.nn.embedding_lookup(embeddings, x)  # (N,D)
    logits = slim.fully_connected(
        inputs=embedded,
        num_outputs=V,
        activation_fn=None
    )
    pred = tf.argmax(logits, axis=-1)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=x_one_hot,
        logits=logits
    )
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                x,
                tf.cast(pred, dtype=x.dtype)
            ),
            dtype=tf.float32
        )
    )
    train_op = slim.learning.create_train_op(
        total_loss=tf.losses.get_total_loss(),
        optimizer=tf.train.AdamOptimizer(learning_rate=3e-5)
    )
    return loss, accuracy, train_op


def main():
    loss, accuracy, train_op = model_fn()
    with tf.train.MonitoredSession() as sess:
        for _ in tqdm(range(I)):
            loss_t, accuracy_t, _ = sess.run([loss, accuracy, train_op])
            tqdm.write("Loss: {:.8f}, Accuracy: {:.5f}".format(loss_t, accuracy_t))
        final_loss, final_accuracy = sess.run([loss, accuracy])
    print("Final Loss: {:.8f}, Final Accuracy: {:.8f}".format(final_loss, final_accuracy))


if __name__ == '__main__':
    main()
