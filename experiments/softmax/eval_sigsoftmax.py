import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

V = 5000
N = 1000
D = 256
I = 10000


def model_fn():
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
    h = tf.nn.embedding_lookup(embeddings, x)  # (N,D)
    h = slim.fully_connected(
        inputs=h,
        num_outputs=D,
        activation_fn=tf.nn.leaky_relu
    )
    h = slim.fully_connected(
        inputs=h,
        num_outputs=D,
        activation_fn=tf.nn.leaky_relu
    )
    h = slim.fully_connected(
        inputs=h,
        num_outputs=D,
        activation_fn=tf.nn.leaky_relu
    )
    logits = slim.fully_connected(
        inputs=h,
        num_outputs=V,
        activation_fn=None
    )
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=x_one_hot,
        logits=logits,
        reduction=tf.losses.Reduction.MEAN
    )
    train_op = slim.learning.create_train_op(
        total_loss=tf.losses.get_total_loss(),
        optimizer=tf.train.AdamOptimizer(learning_rate=3e-5)
    )
    return loss, train_op


def main():
    loss, train_op = model_fn()
    with tf.train.MonitoredSession() as sess:
        for _ in tqdm(range(I)):
            loss_t = sess.run([loss, train_op])[0]
            tqdm.write("Loss: {:.8f}".format(loss_t))
        final_loss = sess.run([loss])[0]
    print("Final Loss: {:.8f}".format(final_loss))


if __name__ == '__main__':
    main()
