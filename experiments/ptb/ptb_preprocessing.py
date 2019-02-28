import math
import os

import numpy as np
import tensorflow as tf

from graph_lm.data.calculate_vocabulary import calculate_vocabulary, calculate_wordmap
from graph_lm.data.treebank import RECORDS, read_treebank_files
from graph_lm.data.write_records import encode_words, write_records


def main(argv):
    sents = list(read_treebank_files(tf.flags.FLAGS.data_dir))
    total = len(sents)

    val_size = int(math.floor(tf.flags.FLAGS.val_size * total))
    test_size = int(math.floor(tf.flags.FLAGS.test_size * total))
    train_size = total - (val_size + test_size)
    print("Records: {} ({} train, {} val, {} test)".format(
        total, train_size, val_size, test_size))

    idx = np.arange(total)
    np.random.shuffle(idx)

    i = 0
    train_idx = idx[i:i + train_size]
    i += train_size
    val_idx = idx[i:i + val_size]
    i += val_size
    test_idx = idx[i:i + test_size]
    i += test_size
    assert i == total

    splits = [train_idx, val_idx, test_idx]
    data_words = [[sents[i] for i in ids] for ids in splits]

    vocab = calculate_vocabulary(data_words)
    vocab = np.array(vocab, dtype=np.unicode_)
    os.makedirs(tf.flags.FLAGS.output_dir, exist_ok=True)
    np.save(os.path.join(tf.flags.FLAGS.output_dir, "vocab.npy"), vocab)
    print("Vocabulary size: {}".format(vocab.size))

    wordmap = calculate_wordmap(vocab)
    data_encoded = [encode_words(sentences, wordmap) for sentences in data_words]
    for f, d in zip(RECORDS, data_encoded):
        write_records(data=d, output_file=os.path.join(tf.flags.FLAGS.output_dir, f))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('data_dir', r'D:\Projects\graph-lm\data\eng_news_txt_tbnk-ptb_revised\data\tokenized_source',
                           'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/eng_news_txt_tbnk-data', 'Data directory')
    tf.flags.DEFINE_float('val_size', .10, 'val_size')
    tf.flags.DEFINE_float('test_size', .10, 'test_size')
    tf.app.run()
