import os

import tensorflow as tf

from .word import ALL_FIELDS, SENTENCE_LENGTH

# from tensorflow.contrib.data.python.ops.batching import map_and_batch
# from tensorflow.contrib.data.python.ops.shuffle_ops import shuffle_and_repeat

TRAIN = 0
VALID = 1
TEST = 2
SPLITS = ['train', 'valid', 'test']
RECORDS = ["{}.tfrecords".format(f) for f in SPLITS]
PARSED = ["{}-parsed.dill".format(s) for s in SPLITS]
VOCAB_FILE = 'vocab.npy'
TAG_FILE = 'taglist.npy'

"""
DEPPARSE_SHAPE = (
    {
        "sequence_length": [],
        'indices': [None],
        'text': [None],
        'tags': [None],
        'heads': [None]
    }, [])
BATCH_SHAPE = (
    {
        "features": [None],
        "feature_length": [],
    }, ([None], []))
"""


def parse_example(serialized_example):
    context_features = {
        SENTENCE_LENGTH: tf.FixedLenFeature([1], tf.int64)
    }
    sequence_features = {
        field: tf.FixedLenSequenceFeature([1], tf.int64)
        for field in ALL_FIELDS
    }
    context_parse, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example, context_features, sequence_features
    )

    features = {
        field: tf.squeeze(sequence_parsed[field], -1) for field in ALL_FIELDS
    }
    features[SENTENCE_LENGTH] = tf.cast(tf.squeeze(context_parse[SENTENCE_LENGTH], -1), tf.int32)

    print(features)
    return features, features[SENTENCE_LENGTH]


def dataset_single(filenames, num_epochs=1, shuffle=True, parse_fn=parse_example):
    ds = tf.data.TFRecordDataset(filenames=filenames)
    if shuffle:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(
            count=num_epochs, buffer_size=tf.flags.FLAGS.shuffle_buffer_size))
    else:
        ds.repeat(count=num_epochs)
    ds = ds.map(
        parse_fn,
        num_parallel_calls=tf.flags.FLAGS.num_parallel_calls)
    return ds


def dataset_batch(ds_single: tf.data.Dataset, batch_shape, batch_size=5):
    ds = ds_single.padded_batch(
        batch_size=batch_size,
        padded_shapes=batch_shape,
        drop_remainder=False)
    return ds


def make_input_fn(data_files, batch_size, shuffle=True, num_epochs=None):
    batch_shape = {
        field: [None] for field in ALL_FIELDS
    }
    batch_shape[SENTENCE_LENGTH] = []

    def input_fn():
        ds = dataset_single(data_files, shuffle=shuffle, num_epochs=num_epochs)
        ds = dataset_batch(ds, batch_size=batch_size, batch_shape=(batch_shape, []))
        ds = ds.prefetch(buffer_size=tf.flags.FLAGS.prefetch_buffer_size)
        return ds

    return input_fn


def make_input_fns(data_dir, batch_size, make_input=make_input_fn):
    train_fn = make_input(
        [os.path.join(data_dir, RECORDS[TRAIN])],
        batch_size=batch_size, shuffle=True, num_epochs=None)
    val_fn = make_input(
        [os.path.join(data_dir, RECORDS[VALID])],
        batch_size=batch_size, shuffle=True, num_epochs=None)
    test_fn = make_input(
        [os.path.join(data_dir, RECORDS[TEST])],
        batch_size=batch_size, shuffle=True, num_epochs=None)
    return train_fn, val_fn, test_fn
