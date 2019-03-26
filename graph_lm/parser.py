import stanfordnlp
import tensorflow as tf

from typing import Callable, Iterable, Union, Optional, List

class Word(object):
    def __init__(self, index, text, head, tag):
        self.index=index
        self.text = text
        self.head = head
        self.tag = tag


def get_pipeline():
    return stanfordnlp.Pipeline(
        lang='en',
        models_dir=tf.flags.FLAGS.stanfordnlp_dir
    )

def parse_docs(docs:Iterable, nlp:stanfordnlp.Pipeline) -> Iterable[List[Word]]:
    for doc in docs:
        parsed = nlp(doc)
        #nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
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

        """
        s = parsed.sentences[0]
        print(s)
        print("Words")
        print(s.words)
        print("tokens")
        print(s.tokens)
        print("dependencies")
        print(s.dependencies)

        print("index")
        print([w.index for w in s.words])
        print([w.text for w in s.words])
        print([w.index for w in s.tokens])
        print("dependencies")
        print()
        print(
        print([w[0].text for w in s.dependencies])
        print([])
        print()
        s.print_dependencies()
        break
        """
