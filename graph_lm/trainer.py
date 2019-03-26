import os

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.training import train_and_evaluate

from .default_params import get_hparams
from .data.inputs import make_input_fns, VOCAB_FILE
from .models.model_vae_binary_tree import make_model_vae_binary_tree
from .models.model_vae_ctc_flat import make_model_vae_ctc_flat
from .models.model_vae_ctc_flat_attn import make_model_vae_ctc_flat_attn
from .models.model_vae_dag import make_model_vae_dag

def make_model_fn(hparams, run_config, vocab):
    if hparams.model == 'vae_binary_tree':
        return make_model_vae_binary_tree(run_config, vocab)
    elif hparams.model == 'vae_ctc_flat':
        return make_model_vae_ctc_flat(run_config, vocab)
    elif hparams.model == 'vae_ctc_flat_attn':
        return make_model_vae_ctc_flat_attn(run_config, vocab)
    elif hparams.model == 'vae_dag':
        return make_model_vae_dag(run_config, vocab)
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

    train_input_fn, eval_input_fn, test_input_fn = make_input_fns(
        tf.flags.FLAGS.data_dir,
        batch_size=tf.flags.FLAGS.batch_size)

    vocab = np.load(os.path.join(tf.flags.FLAGS.data_dir, VOCAB_FILE))

    # Model
    model_fn = make_model_fn(hparams=hparams, run_config=run_config, vocab=vocab)
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
