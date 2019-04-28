import os

import tensorflow as tf

from graph_lm.data.calculate_vocab import calcuate_vocabmap, calculate_characters, write_vocablists
from graph_lm.data.inputs import RECORD_FMT, SPLITS, TEST, TRAIN, VALID, VOCAB_FILE
from graph_lm.data.word import TEXT
from graph_lm.data.readers.ptb_raw import PTB_RAW_SPLITS, read_ptb_characters_fn
from graph_lm.data.write_records import write_records_raw


def main(_argv):
    data_dir = tf.flags.FLAGS.data_dir
    output_dir = tf.flags.FLAGS.output_dir
    data_files = [read_ptb_characters_fn(data_dir, ds) for ds in PTB_RAW_SPLITS]
    data_count = [sum(1 for _ in ds()) for ds in data_files]
    output_fmts = [os.path.join(output_dir, f, RECORD_FMT) for f in SPLITS]
    charset = calculate_characters(data_files[TRAIN]())
    for ds in [VALID, TEST]:
        for sentence in data_files[ds]():
            for char in sentence:
                if char not in charset:
                    raise ValueError("Character [{}] not in charset".format(char))
    charmap = calcuate_vocabmap(charset)
    write_vocablists(
        {TEXT: charset},
        output_dir
    )
    for output_fmt, sentences_fn, total in zip(output_fmts, data_files, data_count):
        write_records_raw(
            sentences=sentences_fn(),
            output_file=output_fmt,
            charmap=charmap,
            chunksize=tf.flags.FLAGS.chunksize,
            total=total
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/ptb/data/ptb-raw', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/ptb/processed-raw', 'Data directory')
    tf.flags.DEFINE_float('min_count', 20, 'min_count')
    tf.flags.DEFINE_integer('chunksize', 1000, 'chunksize')
    tf.app.run()
