"""
new splits from Sid
NLL: 6.2501068115234375
Wordcount: 929435
Sentences: 37947
nll_per_sent: 153.08372497558594
average_length: 24.492977142333984
max_length: 275
assert wordcount: 929435
"""

import tensorflow as tf

from graph_lm.stats import calc_bias


def main(_argv):
    calc_bias(smoothing=0.05)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/ptb/processed-raw', 'Data directory')
    tf.flags.DEFINE_string('data_version', 'v2', 'data_version')
    tf.flags.DEFINE_integer('batch_size', 16, 'Batch size')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('max_steps', 100000, 'max_steps')
    tf.flags.DEFINE_integer('eval_steps', 100, 'max_steps')
    tf.flags.DEFINE_integer('min_after_dequeue', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('prefetch_buffer_size', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('num_parallel_calls', 4, 'min_after_dequeue')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'save_checkpoints_secs')
    tf.app.run()
