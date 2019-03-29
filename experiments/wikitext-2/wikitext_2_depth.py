import os

import tensorflow as tf

from graph_lm.data.read_wikitext import PARSED, TRAIN
from graph_lm.data.write_records import read_records_parsed
from graph_lm.depth import calc_max_depth


def main(_argv):
    data_files = [os.path.join(tf.flags.FLAGS.output_dir, f) for f in PARSED]
    data_parsed = [read_records_parsed(f) for f in data_files]
    depth = calc_max_depth(data_parsed[TRAIN])

    print("Max depth: {}".format(depth))
    # Max depth: 43


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-2', 'Data directory')
    tf.flags.DEFINE_integer('min_count', 20, 'Data directory')
    tf.app.run()
