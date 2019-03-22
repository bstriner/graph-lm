from collections import Counter

UNK = "_UNK_"
BLANK = "_BLANK_"


def calculate_wordmap(vocab):
    return {k: i for i, k in enumerate(vocab)}


def decode_words(ids, vocab):
    return " ".join(vocab[i] if i < len(vocab) else BLANK for i in ids if i >= 0)


def calculate_vocabulary(dataset, min_count=0):
    vocab = Counter()
    for sentence in dataset:
        vocab.update(sentence)
    print("Unique words: {}".format(len(vocab)))
    vocab = list(k for k, count in vocab.items() if count >= min_count)
    vocab.append(UNK)
    vocab.sort()
    return vocab
