

def calculate_wordmap(vocab):
    return {k:i for i, k in enumerate(vocab)}

def decode_words(ids, vocab):
    return " ".join(vocab[i-1] for i in ids if i > 0)

def calculate_vocabulary(datasets):
    vocab = set()
    for dataset in datasets:
        for sentence in dataset:
            for word in sentence:
                vocab.add(word)
    vocab = list(vocab)
    vocab.sort()
    return vocab

