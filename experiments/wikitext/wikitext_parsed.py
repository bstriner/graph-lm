import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

from graph_lm.data.read_wikitext import FILES_RAW, read_wikitext_raw, PARSED
from graph_lm.data.write_records import write_sentences_parsed
from graph_lm.parser import parse_docs, get_pipeline


def main(argv):
    os.makedirs(tf.flags.FLAGS.output_dir, exist_ok=True)
    nlp = get_pipeline()
    data_files = [os.path.join(tf.flags.FLAGS.data_dir, f) for f in FILES_RAW]
    data_docs = [read_wikitext_raw(f) for f in data_files]
    data_parsed = [parse_docs(f, nlp=nlp) for f in data_docs]
    for f, sentences in zip(PARSED, data_parsed):
        write_sentences_parsed(
            sentences=sentences,
            output_file=os.path.join(tf.flags.FLAGS.output_dir, f)
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '/mnt/data/projects/data/wikitext-103-raw', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/wikitext-103', 'Data directory')
    tf.flags.DEFINE_string('stanfordnlp_dir', '/mnt/data/projects/stanfordnlp_resources', 'Data directory')
    tf.flags.DEFINE_integer('min_count', 10, 'Data directory')
    tf.app.run()
