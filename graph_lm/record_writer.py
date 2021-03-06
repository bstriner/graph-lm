
import os

import tensorflow as tf

from graph_lm.data.calculate_vocab import calcuate_vocabmaps, calculate_vocablists, calculate_vocabs, combine_vocabs, \
    write_vocablists
from graph_lm.data.conllu import make_conllu_dataset
from graph_lm.data.inputs import RECORD_FMT, SPLITS
from graph_lm.data.write_records import write_records_parsed_v2

def record_writer(file_names):
    min_counts = {
        'text': tf.flags.FLAGS.min_count,
        'lemma': tf.flags.FLAGS.min_count
    }
    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in file_names]
    output_files = [os.path.join(tf.flags.FLAGS.output_dir, f, RECORD_FMT) for f in SPLITS]
    data_parsed = [make_conllu_dataset(f) for f in data_files]
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