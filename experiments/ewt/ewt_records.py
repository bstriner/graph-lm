import tensorflow as tf

from graph_lm.record_writer import record_writer

EWT_FILES = [
    'en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu', 'en_ewt-ud-test.conllu'
]


def main(_argv):
    record_writer(EWT_FILES)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/ewt/raw', 'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/ewt/processed-small', 'Data directory')
    tf.flags.DEFINE_float('min_count', 20, 'min_count')
    tf.flags.DEFINE_float('max_length', 32, 'max_length')
    tf.flags.DEFINE_integer('chunksize', 1000, 'chunksize')
    tf.app.run()
