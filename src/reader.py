import os
from typing import List, Dict
from collections import Counter

import tensorflow as tf

import constants
from constants import DataSetType


class DataReader:

    def __init__(self):
        self._vocab = None
        self._train = None
        self._test = None
        self._valid = None

        self._train_words = self._read_words(dataset=DataSetType.TRAIN)
        self._valid_words = self._read_words(dataset=DataSetType.VALID)
        self._test_words = self._read_words(dataset=DataSetType.TEST)

    @property
    def vocab(self) -> Dict:
        if self._vocab is not None:
            return self._vocab
        counter = Counter(self._train_words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = list(zip(*count_pairs))
        self._vocab = dict(zip(words, range(len(words))))
        return self._vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def train(self):
        return self._ds_getter(attr='_train', data_set_type=DataSetType.TRAIN)

    @property
    def valid(self):
        return self._ds_getter(attr='_valid', data_set_type=DataSetType.VALID)

    @property
    def test(self):
        return self._ds_getter(attr='_test', data_set_type=DataSetType.TEST)

    @staticmethod
    def _read_words(dataset: DataSetType) -> List[str]:
        file_name = None
        if dataset == DataSetType.TRAIN:
            file_name = constants.TRAIN_FILE_NAME
        elif dataset == DataSetType.TEST:
            file_name = constants.TEST_FILE_NAME
        elif dataset == DataSetType.VALID:
            file_name = constants.VALID_FILE_NAME
        file_path = os.path.join(constants.DATA_PATH, file_name)

        with tf.gfile.GFile(file_path) as f:
            words = f.read().replace('\n', constants.EOS).split()

        return words

    def _ds_getter(self, attr: str, data_set_type: DataSetType) -> List[int]:
        cached = getattr(self, attr, None)
        if cached is not None:
            return cached
        words = self._read_words(data_set_type)
        ds = [self.vocab[w] for w in words if w in self.vocab]
        setattr(self, attr, ds)
        return ds


class BatchProducer:

    def __init__(self, batch_size: int, time_steps: int):
        tf.train.range_input_producer()
