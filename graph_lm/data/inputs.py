import tensorflow as tf
# from tensorflow.contrib.data.python.ops.batching import map_and_batch
from tensorflow.contrib.data.python.ops.shuffle_ops import shuffle_and_repeat


def parse_example(serialized_example):
    context_features = {
        "data_size": tf.FixedLenFeature([1], tf.int64)

    }
    sequence_features = {
        #"data_feat": tf.VarLenFeature(tf.int64)
        "data_feat": tf.FixedLenSequenceFeature([1],tf.int64)

    }

    context_parse, sequence_parsed = tf.parse_single_sequence_example(
        serialized_example, context_features, sequence_features
    )
    feature_length = context_parse['data_size']
    feature_length = tf.squeeze(feature_length, -1)
    feature_length = tf.cast(feature_length, dtype=tf.int32)

    features = sequence_parsed['data_feat']
    features = tf.squeeze(features, -1)
    #print(features)
    #features = tf.sparse_tensor_to_dense(features)
    features = tf.cast(features, dtype=tf.int32)
    features = features + 1
    #features = tf.reshape(features, (-1,))

    feats = {
        "features": features,
        "feature_length": feature_length
    }
    print(feats)
    return feats, (features, feature_length)


def dataset_single(filenames, num_epochs=1, shuffle=True):
    ds = tf.data.TFRecordDataset(filenames=filenames)
    if shuffle:
        ds = ds.apply(shuffle_and_repeat(count=num_epochs, buffer_size=tf.flags.FLAGS.shuffle_buffer_size))
    else:
        ds.repeat(count=num_epochs)
    ds = ds.map(
        parse_example,
        num_parallel_calls=tf.flags.FLAGS.num_parallel_calls)
    return ds


def dataset_batch(ds_single: tf.data.Dataset, batch_size=5):
    feat_shapes = {
        "features": [None],
        "feature_length": [],
    }
    label_shapes = [None], []
    ds = ds_single.padded_batch(
        batch_size=batch_size,
        padded_shapes=(feat_shapes, label_shapes),
        drop_remainder=False)
    return ds


def make_input_fn(data_files, batch_size, shuffle=True, num_epochs=None):
    def input_fn():
        ds = dataset_single(data_files, shuffle=shuffle, num_epochs=num_epochs)
        ds = dataset_batch(ds, batch_size=batch_size)
        ds = ds.prefetch(buffer_size=tf.flags.FLAGS.prefetch_buffer_size)
        return ds

    return input_fn
