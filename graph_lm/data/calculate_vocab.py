import os
from collections import Counter

import numpy as np
from typing import Dict, Iterable, List

from .word import TEXT_FIELDS, Word

UNK = "_UNK_"
BLANK = "_BLANK_"


def calculate_map(vocab):
    return {k: i for i, k in enumerate(vocab)}


def calculate_vocabs(dataset: Iterable[List[Word]], fields: List[str] = TEXT_FIELDS) -> Dict[str, Counter]:
    vocabs = {
        k: Counter() for k in fields
    }
    for sentence in dataset:
        for word in sentence:
            for k in fields:
                vocabs[k].update([getattr(word, k)])
    return vocabs


def combine_vocabs(vocabs: Iterable[Dict[str, Counter]], fields: List[str] = TEXT_FIELDS) -> Dict[str, Counter]:
    vocabs = {
        k: sum((vocab[k] for vocab in vocabs), Counter()) for k in fields
    }
    return vocabs


def calculate_vocablists(vocabs: Dict[str, Counter], min_counts=Dict[str, int]) -> Dict[str, List[str]]:
    return {
        k: list(sorted(
            word for word, count in vocab.items() if k not in min_counts or count >= min_counts[k]
        )) + ([UNK] if k in min_counts and min_counts[k] > 0 else [])
        for k, vocab in vocabs.items()
    }


def calcuate_vocabmaps(vocablists: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    return {
        key: {word: i for i, word in enumerate(vocab)} for key, vocab in vocablists.items()
    }


def write_vocablists(vocablists: Dict[str, List[str]], path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for field, vocab in vocablists.items():
        output_path = os.path.join(path, "{}.npy".format(field))
        np.save(
            file=output_path,
            arr=np.array(vocab, dtype=np.unicode_)
        )


def read_vocablists(path: str, fields: List[str] = TEXT_FIELDS) -> Dict[str, List[str]]:
    return {
        field: np.load(file=os.path.join(path, "{}.npy".format(field))) for field in fields
    }
