import glob
import os
import re


def read_treebank_file(file):
    with open(file, 'r') as f:
        for line in f:
            m = re.match("<en=(\\d*)>(.*)", line)
            if m:
                tokens = m.group(2).split(" ")
                tokens = [t.strip() for t in tokens]
                tokens = [t for t in tokens if len(t) > 0]
                yield tokens


def read_treebank_files(path):
    for file in glob.glob(os.path.join(path, '**', '*.txt'), recursive=True):
        for sentence in read_treebank_file(file):
            yield sentence
