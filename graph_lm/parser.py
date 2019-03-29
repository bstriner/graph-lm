import stanfordnlp
import tensorflow as tf
from stanfordnlp.protobuf.CoreNLP_pb2 import Document
from stanfordnlp.server.client import CoreNLPClient
from tqdm import tqdm
from typing import Generator, Iterable, List

DEPPARSE = "depparse"
ROOT = "root"


class Word(object):
    def __init__(self, index, text, head=0, tag=ROOT):
        self.index = index
        self.text = text
        self.head = head
        self.tag = tag

    def __str__(self):
        return "\"{}\" ({}->{}, {})".format(
            self.text,
            self.index,
            self.head,
            self.tag
        )

    def __repr__(self):
        return str(self)


def get_pipeline():
    return stanfordnlp.Pipeline(
        lang='en',
        models_dir=tf.flags.FLAGS.stanfordnlp_dir
    )


def get_client():
    return CoreNLPClient(
        annotators=[DEPPARSE]
    )


def parse_docs(docs: Iterable, pipeline: stanfordnlp.Pipeline, total=None) -> Generator[List[Word], None, None]:
    for doc in tqdm(docs, desc="Parsing (pipeline)", total=total):
        parsed = pipeline(doc)
        for sentence in parsed.sentences:
            words = [
                Word(
                    index=w[2].index,
                    text=w[2].text,
                    head=w[0].index,
                    tag=w[1]
                )
                for w in sentence.dependencies
            ]
            yield words


def annotate_client(client: CoreNLPClient, doc: str) -> Document:
    return client.annotate(text=doc, annotators=[DEPPARSE])


def parse_docs_client(docs: Iterable, client: CoreNLPClient, count=None) -> Generator[List[Word], None, None]:
    for doc in tqdm(docs, total=count, desc="Parsing (client)"):
        ret = annotate_client(client=client, doc=doc)

        for sentence in ret.sentence:
            print(sentence)
            words = []
            for i, token in enumerate(sentence.token):
                words.append(Word(
                    index=i+1,#token.tokenBeginIndex+1,
                    text=token.word,))
                #tag=token.pos)                )
            for edge in sentence.basicDependencies.edge:
                word = words[edge.target - 1]
                word.tag = edge.dep
                word.head = edge.source
            yield words
