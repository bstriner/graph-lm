from typing import List

from graph_lm.data.parser import Word


def word_depth(sentence: List[Word], index, depth=0):
    word = sentence[index]
    if word.head == 0:
        return depth
    else:
        return word_depth(sentence=sentence, index=word.head - 1, depth=depth + 1)


def calc_depth(sentence: List[Word]):
    depths = (word_depth(sentence=sentence, index=i) for i in range(len(sentence)))
    maxdepth = max(depths)
    return maxdepth


def calc_max_depth(sentences):
    return max(calc_depth(sentence) for sentence in sentences)
