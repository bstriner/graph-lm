import pickle
import gzip
import json
import numpy
import string
import sys
import itertools
import time

from joblib import Parallel, delayed
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

# Define URL of running StanfordCoreNLPServer.
corenlp_url = 'http://localhost:9000'
max_tries = 1000
stanford_corenlp = StanfordCoreNLP(corenlp_url)


def tokenize_and_tag(idx, sentence):
  tries = 0
  while True:
    try:
      annotation = stanford_corenlp.annotate(
        (sentence),
        properties = { 'annotators': 'tokenize,pos,ner,depparse',
                       'outputFormat': 'json' })
      assert type(annotation) == dict
      break
    except Exception:
      time.sleep(1)
      tries += 1
      if tries == 10:
        print("Failed for {}".format(sentence))
        return (idx, None, None, None, None)
      pass
  tokens, pos_tags, ner_tags, depparse = [], [], [], []
  for sentence in annotation['sentences']:
    tokens.append([ token['word'] for token in sentence['tokens'] ])
    pos_tags.append([ token['pos'] for token in sentence['tokens'] ])
    ner_tags.append([ token['ner'] for token in sentence['tokens'] ])
    depparse.append([ (token['dependent'],token['governor']) for token in sentence['basicDependencies'] ])
  return (idx, tokens, pos_tags, ner_tags, depparse)

class Dictionary:
  def __init__(self, lowercase=True, remove_punctuation=True,
               answer_start="ANSWERSTART", answer_end="ANSWEREND"):
    self.index_to_word = []
    self.word_to_index = {}
    self.mutable = True
    self.lowercase = lowercase
    self.remove_punctuation = remove_punctuation
    self.answer_start = answer_start
    self.answer_end = answer_end
    self.pad_index = self.add_or_get_index('<pad>')
    self.pos_tags = dict()
    self.ner_tags = dict()

  def size(self):
    return len(self.index_to_word)

  def add_or_get_index(self, word):
    if word == self.answer_start or word == self.answer_end:
      return -1
    # We ignore punctuation symbols
    if self.remove_punctuation:
      if word in string.punctuation:
        return None
    if self.lowercase:
      word = word.strip().lower()
    if word in self.word_to_index:
      return self.word_to_index[word]
    if not self.mutable:
      return self.pad_index
    new_index = len(self.index_to_word)
    self.word_to_index[word] = new_index
    self.index_to_word.append(word)
    return new_index

  def add_or_get_postag(self, pos_tag):
    if pos_tag in self.pos_tags:
      return self.pos_tags[pos_tag]
    self.pos_tags[pos_tag] = len(self.pos_tags)
    return self.pos_tags[pos_tag]

  def add_or_get_nertag(self, ner_tag):
    if ner_tag in self.ner_tags:
      return self.ner_tags[ner_tag]
    self.ner_tags[ner_tag] = len(self.ner_tags)
    return self.ner_tags[ner_tag]

  def get_index(self, word):
    if word == self.answer_start or word == self.answer_end:
      return -1
    if self.remove_punctuation:
      if word in string.punctuation:
        return None
    if word in self.word_to_index:
      return self.word_to_index[word]
    return self.pad_index

  def get_word(self, index):
    return self.index_to_word[index]

  def set_immutable(self):
    self.mutable = False

class Data:
  def __init__(self, dictionary=None, immutable=False):
    self.dictionary = Dictionary(lowercase=False,
                                 remove_punctuation=False)
    if dictionary:
      self.dictionary = dictionary
    if immutable:
      self.dictionary.set_immutable()
    self.paragraphs = []
    self.tokenized_paras = []
    self.tokenized_para_words = []
    self.paras_pos_tags = []
    self.paras_ner_tags = []
    self.paras_depparse = []
    self.data = []

  def dump_pickle(self, filename):
    with gzip.open(filename + ".gz", 'wb') as fout:
      pickle.dump(self, fout)
      fout.close()

  def read_from_pickle(self, filename):
    with gzip.open(filename + ".gz", 'rb') as fin:
      self = pickle.load(fin)
      fin.close()
      return self

  def get_ids(self, tokenized_text):
    return [ self.dictionary.add_or_get_index(word) \
               for word in tokenized_text ]

  def get_ids_immutable(self, tokenized_text):
    return [ self.dictionary.get_index(word) \
               for word in tokenized_text ]

  ### Assumption every line is a text
  def read_from_file(self, filename, max_articles):
    input_file = open(filename,'r').readlines()
    for para_index, para_text in enumerate(input_file):
      if para_index == max_articles:
        break
      self.paragraphs.append(para_text)
      print("\r{} Paragraphs processed.".format(para_index+1),end="")
      sys.stdout.flush()
    print("\n")

    print("Tokenizing paragraphs ({} total)...".format(len(self.paragraphs)))

    _, self.tokenized_para_words, self.paras_pos_tags, self.paras_ner_tags, self.paras_depparse = \
      zip(*Parallel(n_jobs=-1, verbose=2)(
        delayed(tokenize_and_tag)(None, para_text) for para_text in self.paragraphs))

    self.tokenized_para_words = list(itertools.chain(*self.tokenized_para_words))
    for tokenized_para_words in tqdm(self.tokenized_para_words):
      if tokenized_para_words is None:
        self.tokenized_paras.append(None)
        continue
      self.tokenized_paras.append(self.get_ids(tokenized_para_words))

    self.paras_pos_tags = list(itertools.chain(*self.paras_pos_tags))
    for sent_id, pos_tagged_para in tqdm(enumerate(self.paras_pos_tags)):
      if pos_tagged_para is None:
        continue
      assert len(pos_tagged_para) == len(self.tokenized_paras[sent_id]), str(sent_id)
      self.paras_pos_tags[sent_id] = \
        [ self.dictionary.add_or_get_postag(tag) for tag in pos_tagged_para ]

    self.paras_ner_tags = list(itertools.chain(*self.paras_ner_tags))
    for sent_id, ner_tagged_para in tqdm(enumerate(self.paras_ner_tags)):
      if ner_tagged_para is None:
        continue
      assert len(ner_tagged_para) == len(self.tokenized_paras[sent_id]), str(sent_id)
      self.paras_ner_tags[sent_id] = \
        [ self.dictionary.add_or_get_nertag(tag) for tag in ner_tagged_para ]

    self.paras_depparse = list(itertools.chain(*self.paras_depparse))
    for sent_id, depparse_tagged_para in tqdm(enumerate(self.paras_depparse)):
      if depparse_tagged_para is None:
        continue
      assert len(depparse_tagged_para) == len(self.tokenized_paras[sent_id]), str(sent_id)
      self.paras_depparse[sent_id] = [ t for t in depparse_tagged_para ]

    for sent_id,tokenized_para_words in enumerate(self.tokenized_para_words):
      if tokenized_para_words is not None:
        self.data.append((tokenized_para_words,pos_tagged_para,ner_tagged_para,depparse_tagged_para))
    print("Done.")


# Read train and dev data, either from json files or from pickles, and dump them in
# pickles if necessary.
def read_raw_data(input_raw, max_articles, dump_pickles=None):
  data = Data()
  print("Reading data.")
  data.read_from_file(input_raw, max_articles)
  print("Done.")

  if dump_pickles is not None:
    print("Dumping pickles.")
    data.dump_pickle(dump_pickles)
    print("Done.")

  print("Finished prepping all required data.")
  print("Vocab size is: {}".format(data.dictionary.size()))
  return data

if __name__ == "__main__":
  output=read_raw_data("/project/ocean/sdalmia/projects/pgm/input.txt",10)
  print("len data = {} and output is {}".format(len(output.data),output.data))
