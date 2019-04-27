"""
Train on PTB

ssh -L 6006:localhost:6006 ben@unity
docker run -d -v /mnt/data/projects/graph-lm/output/ptb:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
docker run -d -v /mnt/data/projects/graph-lm/output/ptb/aae:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
"""

import tensorflow as tf

import graph_lm.trainer


def main(_argv):
    graph_lm.trainer.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/ptb/processed', 'Data directory')
    # tf.flags.DEFINE_string('model_dir', '../../output/ptb/aae/binary_tree/flat/v4-bn', 'Data directory')
    tf.flags.DEFINE_string('model_dir', '../../output/ptb/aae/binary_tree/attn/v1', 'Data directory')
    tf.flags.DEFINE_string('config', 'config/aae_binary_tree_attn.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/aae_binary_tree_flat.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_ctc_flat.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_ctc_flat_attn.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_dag.json', 'Data directory')
    tf.flags.DEFINE_integer('batch_size', 8, 'Batch size')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('max_steps', 800000, 'max_steps')
    tf.flags.DEFINE_integer('eval_steps', 100, 'max_steps')
    tf.flags.DEFINE_integer('min_after_dequeue', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('prefetch_buffer_size', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('num_parallel_calls', 4, 'num_parallel_calls')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
