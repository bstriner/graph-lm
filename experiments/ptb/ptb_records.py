import os

import tensorflow as tf

from graph_lm.data.calculate_vocab import calcuate_vocabmaps, calculate_vocablists, calculate_vocabs, combine_vocabs, \
    write_vocablists
from graph_lm.data.conllu import make_conllu_dataset, filter_max_words
from graph_lm.data.inputs import RECORD_FMT, SPLITS
from graph_lm.data.write_records import write_records_parsed_v2
PTB_FILES = [
    'train.conllu', 'dev.conllu', 'test.conllu'
]

"""
Vocab size: 5047
Wrote [29390] records out of [37947]
Wrote [4244] records out of [5490]
Wrote [4223] records out of [5460]
"""


def main(_argv):
    min_counts = {
        'text': tf.flags.FLAGS.min_count,
        'lemma': tf.flags.FLAGS.min_count
    }
    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in PTB_FILES]
    output_files = [os.path.join(tf.flags.FLAGS.output_dir, f, RECORD_FMT) for f in SPLITS]
    data_parsed = [make_conllu_dataset(f, filter_fn=filter_max_words(tf.flags.FLAGS.max_length)) for f in data_files]
    data_count = [sum(1 for _ in ds()) for ds in data_parsed]
    dataset_vocabs = [calculate_vocabs(dataset=ds()) for ds in data_parsed]
    vocabs = combine_vocabs(vocabs=dataset_vocabs)
    vocablists = calculate_vocablists(vocabs, min_counts=min_counts)
    vocabmaps = calcuate_vocabmaps(vocablists)
    write_vocablists(
        vocablists=vocablists,
        path=tf.flags.FLAGS.output_dir
    )
    print("Vocab size: {}".format(len(vocablists['text'])))
    for output_file, sentences, total in zip(output_files, data_parsed, data_count):
        write_records_parsed_v2(
            sentences=sentences(),
            output_file=output_file,
            vocabmaps=vocabmaps,
            total=total,
            chunksize=tf.flags.FLAGS.chunksize,
            max_length=tf.flags.FLAGS.max_length
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/ptb/data/ptb', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/ptb/processed-small', 'Data directory')
    tf.flags.DEFINE_float('min_count', 20, 'min_count')
    tf.flags.DEFINE_float('max_length', 32, 'max_length')
    tf.flags.DEFINE_integer('chunksize', 1000, 'chunksize')
    tf.app.run()
