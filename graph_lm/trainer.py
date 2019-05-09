import os

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate

from .data.calculate_vocab import read_vocablists
from .data.inputs import make_input_fns
from .default_params import get_hparams
from .models.model_binary_tree_attn import make_model_binary_tree_attn
from .models.model_binary_tree_flat import make_model_binary_tree_flat
from .models.model_dag_supervised import make_model_dag_supervised
from .models.model_ctc_flat import make_model_ctc_flat
from .models.model_ctc_flat_attn import make_model_ctc_flat_attn
from .models.model_vae_dag import make_model_dag_unsupervised


def make_model_fn(hparams, run_config, vocabs):
    if hparams.model == 'binary_tree_flat':
        return make_model_binary_tree_flat(run_config, vocabs)
    elif hparams.model == 'binary_tree_attn':
        return make_model_binary_tree_attn(run_config, vocabs)
    elif hparams.model == 'dag_supervised':
        return make_model_dag_supervised(run_config, vocabs)
    elif hparams.model == 'ctc_flat':
        return make_model_ctc_flat(run_config, vocabs)
    elif hparams.model == 'ctc_flat_attn':
        return make_model_ctc_flat_attn(run_config, vocabs)
    elif hparams.model == 'dag_unsupervised':
        return make_model_dag_unsupervised(run_config, vocabs)
    else:
        raise ValueError("Unknown model: {}".format(hparams.model))


def train():
    model_dir = tf.flags.FLAGS.model_dir
    os.makedirs(model_dir, exist_ok=True)
    print("model_dir={}".format(model_dir))
    run_config = RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=tf.flags.FLAGS.save_checkpoints_steps)
    hparams = get_hparams(model_dir, validate=True)
    vocabs = read_vocablists(path=tf.flags.FLAGS.data_dir)
    train_input_fn, eval_input_fn, test_input_fn = make_input_fns(
        data_dir=tf.flags.FLAGS.data_dir,
        batch_size=tf.flags.FLAGS.batch_size,
        data_version=tf.flags.FLAGS.data_version)

    # Model
    model_fn = make_model_fn(hparams=hparams, run_config=run_config, vocabs=vocabs)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=tf.flags.FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=tf.flags.FLAGS.eval_steps, throttle_secs=0)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    train_and_evaluate(
        eval_spec=eval_spec,
        train_spec=train_spec,
        estimator=estimator
    )
