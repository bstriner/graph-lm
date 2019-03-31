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


def read_word(line: str, fmt=None) -> Optional[Word]:
    fmt = fmt or CONLLU_FORMAT
    if len(line.strip()) == 0:
        return None
    else:
        arr = line.split("\t")
        assert len(arr) == len(fmt)
        kwargs = {k: arr[v] for k, v in fmt.items()}
        word = Word(**kwargs)
        return word


def read_conllu(data_file, fmt=None) -> Generator[List[Word], None, None]:
    with open(data_file) as f:
        stack = []
        for line in f:
            word = read_word(line, fmt=fmt)
            if word:
                stack.append(word)
            else:
                yield stack
                stack = []
        if len(stack) > 0:
            yield stack


def make_conllu_dataset(data_file, fmt=None) -> Callable[[], Generator[List[Word], None, None]]:
    def dataset():
        return read_conllu(data_file, fmt=fmt)

    return dataset
