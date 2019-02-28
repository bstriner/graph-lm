import os

import numpy as np
import tensorflow as tf

from graph_lm.data.calculate_vocabulary import calculate_vocabulary
from graph_lm.data.read_wikitext import read_wikitext, FILES


def main(argv):

    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in FILES]
    data_words = [read_wikitext(f) for f in data_files]
    vocab = calculate_vocabulary(data_words)
    vocab = np.array(vocab, dtype=np.unicode_)
    os.makedirs(tf.flags.FLAGS.output_dir, exist_ok=True)
    np.save(os.path.join(tf.flags.FLAGS.output_dir, "vocab.npy"), vocab)
    print("Vocabulary size: {}".format(vocab.size))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', r'D:\Projects\data\wikitext\wikitext-2', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-2', 'Data directory')
    tf.app.run()
