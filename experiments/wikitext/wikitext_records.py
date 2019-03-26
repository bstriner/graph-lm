import os

import numpy as np
import tensorflow as tf

from graph_lm.data.read_wikitext import FILES, read_wikitext, RECORDS
from graph_lm.data.write_records import encode_words, write_records
from graph_lm.data.calculate_vocabulary import calculate_map

def main(argv):
    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in FILES]
    data_words = [read_wikitext(f) for f in data_files]
    vocab = np.load(os.path.join(tf.flags.FLAGS.output_dir, "vocab.npy"))
    wordmap = calculate_map(vocab)
    data_encoded = [encode_words(sentences, wordmap) for sentences in data_words]
    for f, d in zip(RECORDS, data_encoded):
        write_records(data=d, output_file=os.path.join(tf.flags.FLAGS.output_dir, f))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '/mnt/data/projects/data/wikitext-103-raw', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-103', 'Data directory')
    tf.flags.DEFINE_string('stanfordnlp_dir', '/mnt/data/projects/stanfordnlp_resources', 'Data directory')
    tf.app.run()
