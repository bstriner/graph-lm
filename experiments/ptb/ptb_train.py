"""
Train on PTB

ssh -L 6006:localhost:6006 ben@unity
docker run -d -v /mnt/data/projects/graph-lm/output/ptb:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
docker run -d -v /mnt/data/projects/graph-lm/output/ptb/aae:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
docker run -d -v /mnt/data/projects/graph-lm/output/ptb/aae/ctc_flat:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
docker run -d -v /mnt/data/projects/graph-lm/output/ptb/aae/binary_tree:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
"""


import tensorflow as tf

import graph_lm.trainer


"""
import sys
import traceback
import warnings
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback
"""

def main(_argv):
    graph_lm.trainer.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.flags.DEFINE_string('data_dir', '../../data/ptb/processed', 'Data directory')
    tf.flags.DEFINE_string('data_dir', '../../data/ptb/processed-small', 'Data directory')
    tf.flags.DEFINE_string('data_version', 'v1', 'data_version')
    tf.flags.DEFINE_string('model_dir', '../../output/ptb/aae/binary_tree/flat/v15-newds-ae-bn', 'Data directory')
    #tf.flags.DEFINE_string('model_dir', '../../output/ptb/aae/binary_tree/attn/v10', 'Data directory')
    #tf.flags.DEFINE_string('model_dir', '../../output/ptb/aae/dag_supervised/v14', 'Data directory')
    #tf.flags.DEFINE_string('model_dir', '../../output/ptb/aae/ctc_flat/v11-ae', 'Data directory')
    #tf.flags.DEFINE_string('config', 'config/aae_binary_tree_attn_leaves.json', 'Data directory')
    #tf.flags.DEFINE_string('config', 'config/aae_dag_supervised.json', 'Data directory')
    #tf.flags.DEFINE_string('config', 'config/ae_ctc_flat.json', 'Data directory')
    #tf.flags.DEFINE_string('config', 'config/aae_binary_tree_attn.json', 'Data directory')
    tf.flags.DEFINE_string('config', 'config/ae_binary_tree_flat.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_ctc_flat.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_ctc_flat_attn.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_dag.json', 'Data directory')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('max_steps', 800000, 'max_steps')
    tf.flags.DEFINE_integer('eval_steps', 100, 'max_steps')
    tf.flags.DEFINE_bool('gpu_ctc', True, 'gpu_ctc')
    tf.flags.DEFINE_integer('min_after_dequeue', 4000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('shuffle_buffer_size', 4000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('prefetch_buffer_size', 4000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
