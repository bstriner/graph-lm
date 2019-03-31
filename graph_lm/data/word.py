ROOT = "ROOT"
TEXT_FIELDS = [
    'text',
    'lemma',
    'postag1',
    'postag2',
    'cpostag',
    'deprel',
    'phead',
    'pdeprel'
]
INT_FIELDS = [
    'index',
    'head'
]
SENTENCE_LENGTH = 'sentence_length'


class Word(object):
    def __init__(
            self,
            index,
            text,
            lemma=None,
            postag1=ROOT,
            postag2=ROOT,
            cpostag=ROOT,
            head=0,
            deprel=ROOT,
            phead=ROOT,
            pdeprel=ROOT):
        self.index = index
        self.text = text
        self.lemma = lemma if lemma else text
        self.head = head
        self.postag1 = postag1
        self.postag2 = postag2
        self.cpostag = cpostag
        self.deprel = deprel
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        return "\"{}\" ({}->{}, {})".format(
            self.text,
            self.index,
            self.head,
            self.postag1
        )

    def __repr__(self):
        return str(self)
