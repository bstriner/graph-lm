import os

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate

from .default_params import get_hparams
from .kaldi.inputs import make_input_fn
from .models.ctc_model import make_ctc_lstm_model_fn
from .models.ctc_lstm_vae_model import make_ctc_lstm_vae_model_fn


def make_model_fn(hparams, run_config, vocab):
    if hparams.model == 'ctc':
        return make_ctc_lstm_model_fn(run_config, vocab)
    elif hparams.model == 'ctc-vae':
        return make_ctc_lstm_vae_model_fn(run_config, vocab)
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

    # Train Data
    train_input_fn = make_input_fn(
        tf.flags.FLAGS.train_data_dir,
        batch_size=tf.flags.FLAGS.train_batch_size,
        shuffle=True,
        num_epochs=None,
        subsample=hparams.subsample,
        average=False)

    # Test Data
    eval_input_fn = make_input_fn(
        tf.flags.FLAGS.eval_data_dir,
        batch_size=tf.flags.FLAGS.eval_batch_size,
        shuffle=False,
        num_epochs=1,
        subsample=hparams.subsample,
        average=True)

    # Vocab
    vocab = np.load(tf.flags.FLAGS.vocab_file)

    # Model
    model_fn = make_model_fn(hparams=hparams, run_config=run_config, vocab=vocab)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=tf.flags.FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, throttle_secs=0)
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    train_and_evaluate(
        eval_spec=eval_spec,
        train_spec=train_spec,
        estimator=estimator
    )

    """
    elif tf.flags.FLAGS.schedule == 'test':
        train_input_fn, eval_input_fn, test_input_fn, vocab = speech_input_fns(tf.flags.FLAGS.data_dir)
        estimator = Estimator(
            model_fn=make_model_fn(run_config=run_config, vocab=vocab),
            config=run_config,
            params=hparams)
        with open(os.path.join(model_dir, 'test.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Idx', 'True', 'Generated_ML'] + ['Generated_{}'.format(i) for i in range(10)])
            idx = 0
            for x in estimator.predict(
                    input_fn=test_input_fn
            ):
                true = decode_true(x['transcripts'], x['transcript_lengths'], vocab)
                gen = [
                    decode_sparse(z, vocab) for z in x['generated']
                ]
                w.writerow([idx, true] + gen)
                idx += 1
    else:
        print("Unknown schedule")
    """
