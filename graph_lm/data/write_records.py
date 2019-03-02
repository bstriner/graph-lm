import os

import numpy as np
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import Feature, Features
from tensorflow.python.lib.io.tf_record import TFRecordWriter

from .calculate_vocabulary import UNK


def encode_words(sentences, wordmap):
    for sentence in sentences:
        encoded = np.array([wordmap[word] if word in wordmap else wordmap[UNK] for word in sentence], dtype=np.int32)
        yield encoded


def write_records(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with TFRecordWriter(output_file) as writer:
        for datum in data:
            # data_feat, data_size = feature_array(datum.astype(np.int64))
            data_size = feature_int64([datum.shape[0]])
            data_feat = feature_int64_list(datum)
            assert datum.shape[0] > 0
            assert datum.ndim == 1
            sequence_features = {
                'data_feat': data_feat
            }
            context_features = {
                'data_size': data_size
            }
            # example = Example(features=Features(feature=feature))
            example = tf.train.SequenceExample(
                context=Features(feature=context_features),
                feature_lists=tf.train.FeatureLists(feature_list=sequence_features),
            )

            writer.write(example.SerializeToString())


def feature_float32(value):
    return Feature(float_list=tf.train.FloatList(value=value.flatten()))


def feature_int64(value):
    return Feature(int64_list=tf.train.Int64List(value=value))


def feature_int64_list(value):
    return tf.train.FeatureList(feature=[
        Feature(int64_list=tf.train.Int64List(value=[v])) for v in value])


def feature_string(value):
    binary = value.encode('utf-8')
    return Feature(bytes_list=tf.train.BytesList(value=[binary]))


def feature_bytes(value):
    return Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_array(arr):
    shp = np.array(arr.shape, np.int64)
    return feature_bytes(arr.flatten().tobytes()), feature_int64(shp)
