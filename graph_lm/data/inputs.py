import glob
import os

import tensorflow as tf
import itertools
from .word import ALL_FIELDS, SENTENCE_LENGTH

# from tensorflow.contrib.data.python.ops.batching import map_and_batch
# from tensorflow.contrib.data.python.ops.shuffle_ops import shuffle_and_repeat

TRAIN = 0
VALID = 1
TEST = 2
SPLITS = ['train', 'valid', 'test']
RECORD_FMT = "records-{:08d}.tfrecords"
RECORDS = ["{}.tfrecords".format(f) for f in SPLITS]
PARSED = ["{}-parsed.dill".format(s) for s in SPLITS]
VOCAB_FILE = 'vocab.npy'
TAG_FILE = 'taglist.npy'

BATCH_SHAPE = {
    field: [None] for field in ALL_FIELDS
}
BATCH_SHAPE[SENTENCE_LENGTH] = []

BATCH_SHAPE_RAW = {
    'text': [None],
    SENTENCE_LENGTH: []
}


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


def parse_example_raw(serialized_example):
    context_features = {
        SENTENCE_LENGTH: tf.FixedLenFeature([1], tf.int64)
    }
    sequence_features = {
        'text': tf.FixedLenSequenceFeature([1], tf.int64)
    }
    context_parse, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example, context_features, sequence_features
    )

    features = {
        'text': tf.squeeze(sequence_parsed['text'], -1),
        SENTENCE_LENGTH: tf.cast(tf.squeeze(context_parse[SENTENCE_LENGTH], -1), tf.int32)
    }
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


def make_input_fn(data_files, batch_size, shuffle=True,
                  num_epochs=None, data_version="v1"):
    batch_shape = BATCH_SHAPES[data_version]
    parse_fn = PARSE_FNS[data_version]

    def input_fn():
        ds = dataset_single(data_files, shuffle=shuffle, num_epochs=num_epochs, parse_fn=parse_fn)
        ds = dataset_batch(ds, batch_size=batch_size, batch_shape=(batch_shape, []))
        ds = ds.prefetch(buffer_size=tf.flags.FLAGS.prefetch_buffer_size)
        return ds

    return input_fn


def record_files(path):
    return list(glob.glob(os.path.join(path, '**', '*.tfrecords'), recursive=True))


def build_input_dirs(data_dir):
    return [os.path.join(data_dir, split) for split in SPLITS]


def build_input_dir_files(data_dir):
    return [record_files(path) for path in build_input_dirs(data_dir)]


def build_input_files(data_dir):
    return [[os.path.join(data_dir, record_file)] for record_file in RECORDS]


def find_input_files(data_dir):
    files = build_input_files(data_dir)
    if all(os.path.exists(f) and os.path.isfile(f) for f in itertools.chain.from_iterable(files)):
        return files
    dirs = build_input_dirs(data_dir)
    if all(os.path.exists(f) and os.path.isdir(f) for f in dirs):
        return build_input_dir_files(data_dir)
    raise ValueError("Cannot find data files in data_dir: [{}]".format(data_dir))


def make_input_fns(data_dir, batch_size,
                   data_version="v1"):
    data_files = find_input_files(data_dir)
    train_fn = make_input_fn(
        data_files[TRAIN],
        batch_size=batch_size, shuffle=True, num_epochs=None, data_version=data_version)
    val_fn = make_input_fn(
        data_files[VALID],
        batch_size=batch_size, shuffle=True, num_epochs=None, data_version=data_version)
    test_fn = make_input_fn(
        data_files[TEST],
        batch_size=batch_size, shuffle=False, num_epochs=1, data_version=data_version)
    return train_fn, val_fn, test_fn


BATCH_SHAPES = {
    "v1": BATCH_SHAPE,
    "v2": BATCH_SHAPE_RAW
}
PARSE_FNS = {
    "v1": parse_example,
    "v2": parse_example_raw
}