SPLITS = ['train', 'valid', 'test']
FILES = ["wiki.{}.tokens".format(s) for s in SPLITS]
RECORDS = ["{}.tfrecords".format(f) for f in FILES]


def read_wikitext(file):
    for line in open(file, 'r', encoding='UTF-8'):
        for sentence in line.split("."):
            words = sentence.split(" ")
            words = [word.strip() for word in words if len(word.strip()) > 0]
            if len(words) > 0:
                yield words
