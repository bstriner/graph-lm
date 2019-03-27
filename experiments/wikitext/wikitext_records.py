import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

from graph_lm.data.calculate_vocabulary import calculate_map, calculate_vocabulary_and_tags
from graph_lm.data.read_wikitext import RECORDS, TRAIN, PARSED
from graph_lm.data.write_records import write_records_parsed, read_records_parsed


def main(argv):
    data_files = [os.path.join(tf.flags.FLAGS.output_dir, f) for f in PARSED]
    data_parsed = [read_records_parsed(f) for f in data_files]
    vocab, taglist = calculate_vocabulary_and_tags(
        sentences=data_parsed[TRAIN],
        min_count=tf.flags.FLAGS.min_count)
    wordmap = calculate_map(vocab)
    tagmap = calculate_map(taglist)
    np.save(os.path.join(tf.flags.FLAGS.output_dir, "vocab.npy"), vocab)
    np.save(os.path.join(tf.flags.FLAGS.output_dir, "taglist.npy"), taglist)
    for f, sentences in zip(RECORDS, data_parsed):
        write_records_parsed(
            sentences=sentences,
            output_file=os.path.join(tf.flags.FLAGS.output_dir, f),
            wordmap=wordmap,
            tagmap=tagmap
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '/mnt/data/projects/data/wikitext-103-raw', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-103', 'Data directory')
    tf.flags.DEFINE_string('stanfordnlp_dir', '/mnt/data/projects/stanfordnlp_resources', 'Data directory')
    tf.flags.DEFINE_integer('min_count', 10, 'Data directory')
    tf.app.run()
