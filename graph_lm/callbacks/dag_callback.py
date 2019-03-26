import csv
import os

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook

from ..data.calculate_vocabulary import decode_words


class DAGHook(SessionRunHook):
    def __init__(self,
                 true: tf.Tensor, logits: tf.Tensor, idx: tf.Tensor,
                 vocab, path: str, name='Generated'):
        preds = tf.argmax(logits, axis=-1)
        preds_vals = tf.gather_nd(params=preds, indices=idx)
        preds = tf.scatter_nd(
            updates=preds_vals + 1,
            indices=idx,
            shape=tf.cast(tf.shape(preds), dtype=tf.int64)
        ) - 1
        self.preds = preds

        true_vals = tf.gather_nd(params=true, indices=idx)
        true = tf.scatter_nd(
            updates=true_vals + 1,
            indices=idx,
            shape=tf.cast(tf.shape(true), dtype=tf.int64)
        ) - 1
        self.true = true
        self.vocab = vocab
        self.path = path
        self.name = name
        self.step = tf.train.get_or_create_global_step()

    def after_create_session(self, session, coord):
        true, preds, step = session.run([
            self.true,
            self.preds,
            self.step
        ])
        n = true.shape[0]
        output_path = self.path.format(step)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'Row',
                'True',
                self.name])
            for i in range(n):
                tt = decode_words(true[i], vocab=self.vocab)
                tg = decode_words(preds[i], vocab=self.vocab)
                w.writerow([i, tt, tg])
