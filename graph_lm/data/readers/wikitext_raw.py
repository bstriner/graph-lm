from .inputs import SPLITS


WIKITEXT_FILES = ["wiki.{}.tokens".format(s) for s in SPLITS]
WIKITEXT_FILES_RAW = ["wiki.{}.raw".format(s) for s in SPLITS]


def read_wikitext(file):
    for line in open(file, 'r', encoding='UTF-8'):
        for sentence in line.split("."):
            words = sentence.split(" ")
            words = [word.strip() for word in words if len(word.strip()) > 0]
            if len(words) > 0:
                yield words


def read_wikitext_raw(file):
    for line in open(file, 'r', encoding='UTF-8'):
        line = line.strip()
        if (len(line) == 0) or (line[0] == "=" and line[-1] == "="):
            pass
        else:
            yield line


def batch_documents(docs, batch_size=100):
    stack = []
    for doc in docs:
        stack.append(doc)
        if len(stack) == batch_size:
            yield "\n\n".join(stack)
            stack = []
    if len(stack) > 0:
        yield "\n\n".join(stack)
