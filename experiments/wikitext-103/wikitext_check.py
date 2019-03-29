import os

import numpy as np
import tensorflow as tf

from graph_lm.data.calculate_vocabulary import calculate_map
from graph_lm.data.read_wikitext import FILES, read_wikitext, RECORDS, SPLITS
from graph_lm.data.write_records import encode_words, write_records
from graph_lm.data.inputs import make_input_fn
import numpy as np
from graph_lm.data.calculate_vocabulary import calculate_map, decode_words

def main(argv):
    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in RECORDS]
    input_fns = [make_input_fn(
        data_files=[df],
        batch_size=32,
        shuffle=False,
        num_epochs=1
    )().make_one_shot_iterator().get_next() for df in data_files]
    vocab = np.load(os.path.join(tf.flags.FLAGS.data_dir, 'vocab.npy'))
    with tf.train.MonitoredSession() as sess:
        for name, ds in zip(SPLITS, input_fns):
            print(name)
            print(ds)
            tot = 0
            while True:
                features, labels = sess.run(ds)
                print(features)
                print(features['features'])
                print(features['feature_length'])
                for f in features['features']:
                    s = decode_words(f, vocab)
                    print(s)

                raise AttributeError()





if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/wikitext-2', 'Data directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 8, 'Batch size')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('min_after_dequeue', 2000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('grid_size', 10, 'grid_size')
    tf.flags.DEFINE_integer('num_parallel_calls', 4, 'grid_size')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('prefetch_buffer_size', 1000, 'prefetch_buffer_size')

    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
