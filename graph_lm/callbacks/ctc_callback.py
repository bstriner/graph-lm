import csv
import os

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook

from ..data.calculate_vocabulary import decode_words,decode_words_ctc


class CTCHook(SessionRunHook):
    def __init__(self,
                 true: tf.Tensor, logits: tf.Tensor,
                 lengths: tf.Tensor, vocab, path: str, name='Generated', merge_repeated=False):
        self.true = true
        self.logits = logits
        self.lengths = lengths
        self.vocab = vocab
        self.path = path
        self.name = name

        (self.generated_sparse,), _ = tf.nn.ctc_beam_search_decoder_v2(
            inputs=logits,
            sequence_length=lengths,
            #merge_repeated=merge_repeated,
            top_paths=1
        )
        self.generated = tf.sparse_tensor_to_dense(self.generated_sparse, default_value=-1)
        self.generated_raw = tf.transpose(tf.argmax(logits, axis=-1), (1, 0))
        self.step = tf.train.get_or_create_global_step()

    def after_create_session(self, session, coord):
        true, gen, gen_raw, step = session.run([
            self.true,
            self.generated,
            self.generated_raw,
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
                self.name,
                '{} Raw'.format(self.name)])
            for i in range(n):
                tt = decode_words_ctc(true[i], vocab=self.vocab)
                tg = decode_words_ctc(gen[i], vocab=self.vocab)
                tgr = decode_words_ctc(gen_raw[i], vocab=self.vocab)
                w.writerow([i, tt, tg, tgr])
