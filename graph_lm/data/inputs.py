import os

import tensorflow as tf

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


def parse_depparse_example(serialized_example):
    context_features = {
        "data_size": tf.FixedLenFeature([1], tf.int64)

    }
    sequence_features = {
        'indices': tf.FixedLenSequenceFeature([1], tf.int64),
        'text': tf.FixedLenSequenceFeature([1], tf.int64),
        'tags': tf.FixedLenSequenceFeature([1], tf.int64),
        'heads': tf.FixedLenSequenceFeature([1], tf.int64)
    }

    context_parse, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example, context_features, sequence_features
    )
    sequence_length = context_parse['data_size']
    sequence_length = tf.squeeze(sequence_length, -1)
    # sequence_length = tf.cast(sequence_length, dtype=tf.int32)

    features = {
        k: tf.squeeze(sequence_parsed[k], -1) for k in sequence_features.keys()
    }
    features['sequence_length'] = sequence_length
    return features, sequence_length


def parse_example(serialized_example):
    context_features = {
        "data_size": tf.FixedLenFeature([1], tf.int64)

    }
    sequence_features = {
        # "data_feat": tf.VarLenFeature(tf.int64)
        "data_feat": tf.FixedLenSequenceFeature([1], tf.int64)

    }

    context_parse, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example, context_features, sequence_features
    )
    feature_length = context_parse['data_size']
    feature_length = tf.squeeze(feature_length, -1)
    feature_length = tf.cast(feature_length, dtype=tf.int32)

    features = sequence_parsed['data_feat']
    features = tf.squeeze(features, -1)
    # print(features)
    # features = tf.sparse_tensor_to_dense(features)
    features = tf.cast(features, dtype=tf.int32)
    # features = features + 1
    # features = tf.reshape(features, (-1,))

    feats = {
        "text": features,
        "sequence_length": feature_length
    }
    print(feats)
    return feats, (features, feature_length)


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
    def input_fn():
        ds = dataset_single(data_files, shuffle=shuffle, num_epochs=num_epochs)
        ds = dataset_batch(ds, batch_size=batch_size, batch_shape=BATCH_SHAPE)
        ds = ds.prefetch(buffer_size=tf.flags.FLAGS.prefetch_buffer_size)
        return ds

    return input_fn


def make_input_depparse_fn(data_files, batch_size, shuffle=True, num_epochs=None):
    def input_fn():
        ds = dataset_single(data_files, shuffle=shuffle, num_epochs=num_epochs, parse_fn=parse_depparse_example)
        ds = dataset_batch(ds, batch_size=batch_size, batch_shape=DEPPARSE_SHAPE)
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


def make_input_depparse_fns(data_dir, batch_size):
    return make_input_fns(data_dir=data_dir, batch_size=batch_size, make_input=make_input_depparse_fn)
