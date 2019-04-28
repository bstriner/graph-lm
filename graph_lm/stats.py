import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .data.calculate_vocab import read_vocablists, write_vocablist
from .data.inputs import RECORDS, TRAIN, make_input_fn, make_input_fns, find_input_files, PARSE_FNS, BATCH_SHAPES
from .data.word import SENTENCE_LENGTH, TEXT
TALLY_FILE = 'tally.npz'


def bias_ops(ds: tf.data.Dataset, V):
    features, labels = ds.make_one_shot_iterator().get_next()
    tokens = features[TEXT]  # (N, L)
    token_lengths = features[SENTENCE_LENGTH]  # (N,)
    vocab_tally = tf.get_local_variable(
        name='vocab_tally',
        dtype=tf.int64,
        initializer=tf.initializers.zeros,
        shape=(V,)
    )  # (V,)
    word_count = tf.get_local_variable(
        name='word_count',
        dtype=token_lengths.dtype,
        initializer=tf.initializers.zeros,
        shape=[]
    )
    max_length = tf.get_local_variable(
        name='max_length',
        dtype=token_lengths.dtype,
        initializer=tf.initializers.zeros,
        shape=[]
    )
    sentence_count = tf.get_local_variable(
        name='sentence_count',
        dtype=tf.int32,
        initializer=tf.initializers.zeros,
        shape=[]
    )
    mask = tf.sequence_mask(
        maxlen=tf.shape(tokens)[1],
        lengths=token_lengths
    )  # (N, L)
    valid_tokens = tf.boolean_mask(tensor=tokens, mask=mask)  # (Z,)
    update_tally = tf.scatter_nd_add(
        ref=vocab_tally,
        indices=tf.expand_dims(valid_tokens, 1),
        updates=tf.ones(shape=tf.shape(valid_tokens), dtype=vocab_tally.dtype)
    )
    update_sentence_count = tf.assign_add(ref=sentence_count, value=tf.shape(tokens)[0])
    update_word_count = tf.assign_add(ref=word_count, value=tf.reduce_sum(token_lengths))
    update_max_length = tf.assign(ref=max_length, value=tf.maximum(
        max_length,
        tf.reduce_max(token_lengths)
    ))
    update = tf.group(update_tally, update_sentence_count, update_word_count, update_max_length)
    return vocab_tally, sentence_count, word_count, max_length, update


def calc_bias(smoothing=0.01):
    # vocab = np.load(os.path.join(tf.flags.FLAGS.data_dir, VOCAB_FILE))
    data_version = tf.flags.FLAGS.data_version
    vocabs = read_vocablists(path=tf.flags.FLAGS.data_dir, fields=[TEXT])
    vocab=vocabs[TEXT]
    V = vocab.shape[0]
    data_files = find_input_files(tf.flags.FLAGS.data_dir)
    train_input_fn = make_input_fn(
        data_files=data_files[TRAIN],
        batch_size=tf.flags.FLAGS.batch_size,
        shuffle=False,
        num_epochs=1,
        data_version=data_version)
    train_ds = train_input_fn()
    tally_op, sents_op, words_op, max_length_op, update = bias_ops(ds=train_ds, V=V)

    with tf.train.MonitoredSession() as sess:
        it = tqdm(desc='Counting Vocabulary')
        try:
            while True:
                sess.run(update)
                it.update(1)
        except tf.errors.OutOfRangeError:
            pass
        it.close()
        tally, sents, words, max_length = sess.run([tally_op, sents_op, words_op, max_length_op])

    p = tally.astype(np.float32)
    p = p / np.sum(p)
    p = ((1 - smoothing) * p) + ((1. / V) * smoothing)
    nll = -np.sum(np.log(p) * p)
    print("NLL: {}".format(nll))
    print("Wordcount: {}".format(words))
    print("Sentences: {}".format(sents))
    average_length = words.astype(np.float32) / sents.astype(np.float32)
    nll_per_sent = nll * average_length
    print("nll_per_sent: {}".format(nll_per_sent))
    print("average_length: {}".format(average_length))
    print("max_length: {}".format(max_length))
    assert_wordcount = np.sum(tally)
    print("assert wordcount: {}".format(assert_wordcount))
    print("vocab size: {}".format(V))
    assert assert_wordcount == words

    np.savez(os.path.join(tf.flags.FLAGS.data_dir, TALLY_FILE),
             tally=tally,
             average_length=average_length)


def get_bias(smoothing=0.01):
    data = np.load(os.path.join(tf.flags.FLAGS.data_dir, TALLY_FILE))
    tally = data['tally']
    V = tally.shape[0]
    p = tally.astype(np.float32)
    p = p / np.sum(p)
    if smoothing > 0:
        p = ((1 - smoothing) * p) + ((1. / V) * smoothing)
    bias = np.log(p)
    bias = bias - np.max(bias)
    return bias


def get_bias_ctc(average_output_length, smoothing=0.01):
    data = np.load(os.path.join(tf.flags.FLAGS.data_dir, TALLY_FILE))
    tally = data['tally']
    average_length = data['average_length']

    V = tally.shape[0]
    p = tally.astype(np.float32)
    p = p / np.sum(p)
    if smoothing > 0:
        p = ((1 - smoothing) * p) + ((1. / V) * smoothing)
    p_blank = 1. - (average_length / average_output_length)
    assert p_blank > 0
    assert p_blank < 1
    print("p_blank: {}".format(p_blank))
    p_all = np.concatenate([p * (1 - p_blank), [p_blank]], axis=0)
    bias = np.log(p_all)
    bias = bias - np.max(bias)
    return bias
