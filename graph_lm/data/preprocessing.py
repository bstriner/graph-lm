import math
import os

import numpy as np

from graph_lm.data.calculate_vocabulary import calculate_vocabulary, calculate_wordmap
from graph_lm.data.inputs import RECORDS
from graph_lm.data.write_records import encode_words, write_records


def generate_splits(sents, val_ratio, test_ratio):
    total = len(sents)

    val_size = int(math.floor(val_ratio * total))
    test_size = int(math.floor(test_ratio * total))
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
    return data_words


def preprocessing(data_words, output_dir, min_count=0):
    vocab = calculate_vocabulary(data_words[0], min_count=min_count)
    vocab = np.array(vocab, dtype=np.unicode_)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "vocab.npy"), vocab)
    print("Vocabulary size: {}".format(vocab.size))

    wordmap = calculate_wordmap(vocab)
    data_encoded = [encode_words(sentences, wordmap) for sentences in data_words]
    for f, d in zip(RECORDS, data_encoded):
        write_records(data=d, output_file=os.path.join(output_dir, f))
