"""

NLL: 6.125174522399902
Wordcount: 965000
Sentences: 39367
nll_per_sent: 150.14588928222656
average_length: 24.512916564941406
max_length: 275
assert wordcount: 965000
"""

import tensorflow as tf

from graph_lm.data.inputs import make_input_depparse_fn
from graph_lm.stats import calc_bias


def main(_argv):
    calc_bias(smoothing=0.05, input_fn=make_input_depparse_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/wikitext-2', 'Data directory')
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
