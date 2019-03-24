"""
Train on PTB

ssh -L 6006:localhost:6006 ben@unity
docker run -it -v /mnt/data/projects/graph-lm/output:/tb -p 6006:6006 tensorflow/tensorflow tensorboard --logdir /tb
"""

import tensorflow as tf

import graph_lm.trainer


def main(argv):
    graph_lm.trainer.train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', '../../data/eng_news_txt_tbnk-data', 'Data directory')
    tf.flags.DEFINE_string('model_dir', '../../output/ptb/flat-attn/v27-nokl-anneal-1e-4', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_binary_tree.json', 'Data directory')
    # tf.flags.DEFINE_string('config', 'config/vae_ctc_flat.json', 'Data directory')
    tf.flags.DEFINE_string('config', 'config/vae_ctc_flat_attn.json', 'Data directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.flags.DEFINE_integer('capacity', 4000, 'capacity')
    tf.flags.DEFINE_integer('max_steps', 100000, 'max_steps')
    tf.flags.DEFINE_integer('eval_steps', 100, 'max_steps')
    tf.flags.DEFINE_integer('min_after_dequeue', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('shuffle_buffer_size', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('prefetch_buffer_size', 1000, 'min_after_dequeue')
    tf.flags.DEFINE_integer('num_parallel_calls', 4, 'min_after_dequeue')
    tf.flags.DEFINE_integer('grid_size', 10, 'grid_size')
    tf.flags.DEFINE_integer('queue_threads', 2, 'queue_threads')
    tf.flags.DEFINE_integer('save_checkpoints_steps', 2000, 'save_checkpoints_secs')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
