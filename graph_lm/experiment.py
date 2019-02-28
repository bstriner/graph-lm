import tensorflow as tf
from tensorflow.contrib.learn import Experiment
from tensorflow.python.estimator.estimator import Estimator

from .speech_input import speech_input_fns


def make_experiment_fn(make_model_fn):
    def experiment_fn(run_config, hparams):
        train_input_fn, eval_input_fn, test_input_fn, vocab = speech_input_fns(tf.flags.FLAGS.data_dir)
        estimator = Estimator(
            model_fn=make_model_fn(run_config=run_config, vocab=vocab),
            config=run_config,
            params=hparams)
        experiment = Experiment(
            estimator=estimator,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn
        )

        return experiment

    return experiment_fn
