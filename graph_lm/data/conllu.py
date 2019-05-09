from typing import Callable, Generator, List, Optional

from .word import Word

CONLLU_FORMAT = {
    "index": 0,
    "text": 1,
    "lemma": 2,
    "postag1": 3,
    "postag2": 4,
    "cpostag": 5,
    "head": 6,
    "deprel": 7,
    "phead": 8,
    "pdeprel": 9
}

EOS = object()


def read_word(line: str, fmt=None) -> Optional[Word]:
    fmt = fmt or CONLLU_FORMAT
    if len(line.strip()) == 0:
        return EOS
    elif line[0] == "#":
        return None
    else:
        arr = line.split("\t")
        assert len(arr) == len(fmt)
        kwargs = {k: arr[v] for k, v in fmt.items()}
        word = Word(**kwargs)
        try:
            _ = int(arr[0])
            return word
        except ValueError:
            pass


def read_conllu(data_file, fmt=None) -> Generator[List[Word], None, None]:
    with open(data_file, encoding='utf-8') as f:
        stack = []
        for line in f:
            word = read_word(line, fmt=fmt)
            if word == EOS:
                yield stack
                stack = []
            else:
                if word:
                    stack.append(word)
        if len(stack) > 0:
            yield stack


def filter_max_words(max_words):
    def filter_fn(ds):
        for sentence in ds:
            if len(sentence) <= max_words:
                yield sentence

    return filter_fn


def make_conllu_dataset(data_file, fmt=None, filter_fn=None) -> Callable[[], Generator[List[Word], None, None]]:
    def dataset():
        ds = read_conllu(data_file, fmt=fmt)
        if filter_fn is not None:
            ds = filter_fn(ds)
        return ds

    return dataset
