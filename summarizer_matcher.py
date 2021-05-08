import argparse
import numpy as np
import pandas as pd

import os
import json
import gzip

import csv
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

tokenizer = None

def get_tokenizer(lm='bert'):
    """Return the tokenizer. Intiailize it if not initialized.

    Args:
        lm (string): the name of the language model (bert, albert, or distilbert)
    Returns:
        BertTokenizer or DistilBertTokenizer or AlbertTokenizer
    """
    global tokenizer
    if tokenizer is None:
        if lm == 'bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif lm == 'distilbert':
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif lm == 'albert':
            from transformers import AlbertTokenizer
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        elif lm == 'roberta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif lm == 'xlnet':
            from transformers import XLNetTokenizer
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        elif lm == 'longformer':
            from transformers import LongformerTokenizer
            tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    return tokenizer

class Summarizer:
    """To summarize a data entry pair into length up to the max sequence length.

    Args:
        task_config (Dictionary): the task configuration
        lm (string): the language model (bert, albert, or distilbert)

    Attributes:
        config (Dictionary): the task configuration
        tokenizer (Tokenizer): a tokenizer from the huggingface library
    """
    def __init__(self, task_config, lm):
        self.config = task_config
        self.tokenizer = get_tokenizer(lm=lm)
        self.len_cache = {}

        # build the tfidf index
        self.build_index()

    def build_index(self):
        """Build the idf index.

        Store the index and vocabulary in self.idf and self.vocab.
        """
        fns = [self.config['trainset'],
               self.config['validset'],
               self.config['testset']]
        content = []
        for fn in fns:
            with open(fn) as fin:
                for line in fin:
                    LL = line.split('\t')
                    if len(LL) > 2:
                        for entry in LL:
                            content.append(entry)

        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_

    def get_len(self, word):
        """Return the sentence_piece length of a token.
        """
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length

    def transform(self, row, max_len=128):
        """Summarize one single example.

        Only retain tokens of the highest tf-idf

        Args:
            row (str): a matching example of two data entries and a binary label, separated by tab
            max_len (int, optional): the maximum sequence length to be summarized to

        Returns:
            str: the summarized example
        """
        try:
            sentA, sentB, label = row.strip().split('\t')
        except:
            print(row)
            return
        res = ''
        cnt = Counter()
        for sent in [sentA, sentB]:
            tokens = sent.split(' ')
            for token in tokens:
                if token not in ['COL', 'VAL'] and \
                   token not in stop_words:
                    if token in self.vocab:
                        cnt[token] += self.idf[self.vocab[token]]

        for sent in [sentA, sentB]:
            token_cnt = Counter(sent.split(' '))
            total_len = token_cnt['COL'] + token_cnt['VAL']

            subset = Counter()
            for token in set(token_cnt.keys()):
                subset[token] = cnt[token]
            subset = subset.most_common(max_len)

            topk_tokens_copy = set([])
            for word, _ in subset:
                bert_len = self.get_len(word)
                if total_len + bert_len > max_len:
                    break
                total_len += bert_len
                topk_tokens_copy.add(word)

            num_tokens = 0
            for token in sent.split(' '):
                if token in ['COL', 'VAL']:
                    res += token + ' '
                elif token in topk_tokens_copy:
                    res += token + ' '
                    topk_tokens_copy.remove(token)

            res += '\t'

        res += label + '\n'
        return res

    def transform_file(self, input_fn, max_len=256, overwrite=False):
        """Summarize all lines of a tsv file.

        Run the summarizer. If the output already exists, just return the file name.

        Args:
            input_fn (str): the input file name
            max_len (int, optional): the max sequence len
            overwrite (bool, optional): if true, then overwrite any cached output

        Returns:
            str: the output file name
        """
        out_fn = input_fn + '.su'
        if not os.path.exists(out_fn) or \
           os.stat(out_fn).st_size == 0 or overwrite:
            with open(out_fn, 'w') as fout:
                for line in open(input_fn):
                    fout.write(self.transform(line, max_len=max_len))
        return out_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/Beer')
    parser.add_argument("--input_path", type=str, default='input/candidates_small.jsonl')
    parser.add_argument("--output_path", type=str, default='output/matched_small.jsonl')
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--max_len", type=int, default=256)
    hp = parser.parse_args()

    # load the models
    config = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    trainset = hp.input_path
    summarizer = Summarizer(config, hp.lm)
    summarizer.transform_file(trainset, max_len=hp.max_len)
