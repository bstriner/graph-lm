import os

import numpy as np
import tensorflow as tf

from graph_lm.data.calculate_vocabulary import calculate_map, calculate_vocabulary_and_tags
from graph_lm.data.read_wikitext import PARSED, RECORDS, TRAIN
from graph_lm.data.write_records import read_records_parsed, write_records_parsed


def main(_argv):
    data_files = [os.path.join(tf.flags.FLAGS.output_dir, f) for f in PARSED]
    data_parsed = [read_records_parsed(f) for f in data_files]
    data_count = [sum(1 for _ in read_records_parsed(f)) for f in data_files]
    vocab, taglist = calculate_vocabulary_and_tags(
        sentences=read_records_parsed(data_files[TRAIN]),
        total=data_count[TRAIN],
        min_count=tf.flags.FLAGS.min_count)
    wordmap = calculate_map(vocab)
    tagmap = calculate_map(taglist)
    np.save(os.path.join(tf.flags.FLAGS.output_dir, "vocab.npy"), vocab)
    np.save(os.path.join(tf.flags.FLAGS.output_dir, "taglist.npy"), taglist)
    for f, sentences, total in zip(RECORDS, data_parsed, data_count):
        write_records_parsed(
            sentences=sentences,
            output_file=os.path.join(tf.flags.FLAGS.output_dir, f),
            wordmap=wordmap,
            tagmap=tagmap,
            total=total
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-2', 'Data directory')
    tf.flags.DEFINE_integer('min_count', 20, 'Data directory')
    tf.app.run()
