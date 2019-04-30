import os

import dill
import numpy as np
import tensorflow as tf
from tensorflow.core.example.feature_pb2 import Feature, Features
from tensorflow.python.lib.io.tf_record import TFRecordWriter
from tqdm import tqdm
from typing import Dict, Generator, Iterable, List

from .calculate_vocabulary import UNK
from .record_writer import ShardRecordWriter
from .word import INT_FIELDS, SENTENCE_LENGTH, TEXT_FIELDS, Word


def encode_words(words, wordmap):
    return np.array([wordmap[word] if word in wordmap else wordmap[UNK] for word in words], dtype=np.int64)


def encode_sentences(sentences, wordmap):
    for sentence in sentences:
        yield encode_words(sentence, wordmap)


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


def write_sentences_parsed(sentences: Iterable[List[Word]], output_file):
    with open(output_file, 'wb') as f:
        for sentence in sentences:
            dill.dump(sentence, f)


def read_records_parsed(parsed_file) -> Generator[List[Word], None, None]:
    with open(parsed_file, 'rb') as f:
        try:
            while True:
                yield dill.load(f)
        except EOFError:
            pass


def write_records_parsed(sentences: Iterable[List[Word]], output_file, wordmap, tagmap, total=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with TFRecordWriter(output_file) as writer:
        for sentence in tqdm(sentences, desc="Writing Records", total=total):
            indices = [word.index for word in sentence]
            text = [wordmap[word.text] if word.text in wordmap else wordmap[UNK] for word in sentence]
            tags = [tagmap[word.tag] if word.tag in tagmap else tagmap[UNK] for word in sentence]
            heads = [word.head for word in sentence]

            indices_feat = feature_int64_list(indices)
            text_feat = feature_int64_list(text)
            tags_feat = feature_int64_list(tags)
            heads_feat = feature_int64_list(heads)
            data_size = feature_int64([len(sentence)])
            sequence_features = {
                'indices': indices_feat,
                'text': text_feat,
                'tags': tags_feat,
                'heads': heads_feat
            }
            context_features = {
                'data_size': data_size
            }
            example = tf.train.SequenceExample(
                context=Features(feature=context_features),
                feature_lists=tf.train.FeatureLists(feature_list=sequence_features),
            )
            writer.write(example.SerializeToString())


def write_records_parsed_v2(
        sentences: Iterable[List[Word]],
        output_file: str,
        vocabmaps: Dict[str, Dict[str, int]],
        int_fields=INT_FIELDS,
        text_fields=TEXT_FIELDS,
        chunksize=1000,
max_length=None,
        total=None):
    count = 0
    with ShardRecordWriter(path_fmt=output_file, chunksize=chunksize) as writer:
        for sentence in tqdm(sentences, desc="Writing Records", total=total):
            if max_length is None or len(sentence) <= max_length:
                count += 1
                int_field_data = {
                    field: feature_int64_list([int(getattr(word, field)) for word in sentence])
                    for field in int_fields}
                text_field_data = {
                    field: feature_int64_list(encode_words(
                        words=[getattr(word, field) for word in sentence],
                        wordmap=vocabmaps[field]))
                    for field in text_fields}

                sentence_length = feature_int64([len(sentence)])
                sequence_features = dict()
                sequence_features.update(int_field_data)
                sequence_features.update(text_field_data)
                context_features = {
                    SENTENCE_LENGTH: sentence_length
                }
                example = tf.train.SequenceExample(
                    context=Features(feature=context_features),
                    feature_lists=tf.train.FeatureLists(feature_list=sequence_features),
                )
                writer.write(example.SerializeToString())
    print("Wrote [{}] records out of [{}]".format(count, total))

def write_records_raw(sentences, output_file, charmap, chunksize=1000, total=None):
    with ShardRecordWriter(path_fmt=output_file, chunksize=chunksize) as writer:
        for sentence in tqdm(sentences, total=total, desc='Writing Records'):
            datum = encode_words(sentence, charmap)
            data_size = feature_int64([datum.shape[0]])
            data_feat = feature_int64_list(datum)
            assert datum.shape[0] > 0
            assert datum.ndim == 1
            sequence_features = {
                'text': data_feat
            }
            context_features = {
                SENTENCE_LENGTH: data_size
            }
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
