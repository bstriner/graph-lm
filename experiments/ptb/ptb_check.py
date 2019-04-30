import tensorflow as tf

from graph_lm.data.treebank import read_treebank_files
from graph_lm.data.inputs import make_input_fns, TRAIN

def main(argv):
    fns = make_input_fns(
        data_dir=tf.flags.FLAGS.data_dir,
        data_version=tf.flags.FLAGS.data_version,
        batch_size=tf.flags.FLAGS.batch_size
    )
    train_fn = fns[TRAIN]
    ds = train_fn().make_one_shot_iterator()
    feats, labels = ds.get_next()
    print(feats)
    with tf.train.MonitoredSession() as sess:
        text, head, length = sess.run([feats['text'], feats['head'], feats['sentence_length']])
        print(text[0])
        print(head[0])
        print(length[0])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/ptb/processed', 'Data directory')
    tf.flags.DEFINE_string('data_version', 'v1', 'data_version')
    tf.flags.DEFINE_integer('batch_size', 32, 'batch_size')
    tf.flags.DEFINE_integer('shuffle_buffer_size', 4000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('prefetch_buffer_size', 4000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.app.run()
