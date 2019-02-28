import os

import numpy as np
import tensorflow as tf

from graph_lm.data.read_wikitext import FILES, read_wikitext, RECORDS
from graph_lm.data.write_records import encode_words, write_records
from graph_lm.data.calculate_vocabulary import calculate_wordmap

def main(argv):
    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in FILES]
    data_words = [read_wikitext(f) for f in data_files]
    vocab = np.load(os.path.join(tf.flags.FLAGS.output_dir, "vocab.npy"))
    wordmap = calculate_wordmap(vocab)
    data_encoded = [encode_words(sentences, wordmap) for sentences in data_words]
    for f, d in zip(RECORDS, data_encoded):
        write_records(data=d, output_file=os.path.join(tf.flags.FLAGS.output_dir, f))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', r'D:\Projects\data\wikitext\wikitext-2', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-2', 'Data directory')
    tf.app.run()
