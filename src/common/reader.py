import os
import csv
from typing import List
from collections import Counter

import numpy as np
import tensorflow as tf

from common import constants


def read_words_from_file(file_name: 'str') -> 'List[str]':
    file_path = os.path.join(constants.DATA_PATH, file_name)
    with tf.gfile.GFile(file_path) as f:
        words = f.read().replace('\n', constants.EOS).split()
    words.append(constants.UNK)
    return words


class Vocab(dict):

    @classmethod
    def build(cls, file_name: 'str') -> 'Vocab':
        data = read_words_from_file(file_name)
        counter = Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = list(zip(*count_pairs))
        return cls(zip(words, range(len(words))))

    @property
    def size(self) -> 'int':
        return len(self)

    def map_words(self, words: 'List[str]') -> 'List[int]':
        unk_index = self[constants.UNK]
        return [self.get(w, unk_index) for w in words]


def read_train_data(file_name: 'str', vocab: 'Vocab') -> 'np.array':
    words = read_words_from_file(file_name)
    return np.array(vocab.map_words(words))


def read_evaluation_data(file_name: 'str', vocab: 'Vocab') -> ('np.array', 'np.array', 'np.array'):
    file_path = os.path.join(constants.DATA_PATH, file_name)
    x1s, x2s, ys = [], [], []
    with open(file_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            x1s.append(row[0])
            x2s.append(row[1])
            ys.append(float(row[2]))
    return np.array(vocab.map_words(x1s)), np.array(vocab.map_words(x2s)), np.array(ys)
