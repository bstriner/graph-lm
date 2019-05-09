import tensorflow as tf

from graph_lm.record_writer import record_writer

EWT_FILES = [
    'en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu', 'en_ewt-ud-test.conllu'
]

"""
Writing Records:   0%|          | 0/12543 [00:00<?, ?it/s]Vocab size: 1334
Writing Records: 100%|██████████| 12543/12543 [00:19<00:00, 639.63it/s]
Wrote [11325] records out of [12543]
Writing Records: 100%|██████████| 2002/2002 [00:02<00:00, 703.06it/s]
Wrote [1890] records out of [2002]
Writing Records: 100%|██████████| 2077/2077 [00:02<00:00, 804.34it/s]
Wrote [1971] records out of [2077]
"""
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
