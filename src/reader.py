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

    def __init__(self, raw_data: List[int], batch_size: int, time_steps: int, name=None):
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.batch_shape = [batch_size, time_steps]

        raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)
        num_batches = tf.size(raw_data) // batch_size
        self.data = tf.reshape(raw_data[:num_batches * batch_size], [batch_size, num_batches])

        epoch_size = (num_batches - 1) // time_steps
        assertion = tf.assert_positive(epoch_size, message='epoch_size == 0, which is not appropriate')
        with tf.control_dependencies([assertion]):
            self.epoch_size = tf.identity(epoch_size, name='epoch_size')

        self.epoch_queue = tf.train.range_input_producer(epoch_size, shuffle=False)

    def get_init_op(self):
        return self.epoch_queue

    def build_batch(self):
        batch_index = self.epoch_queue.dequeue()

        batch_index = tf.Print(batch_index, [batch_index])

        x_batch = tf.strided_slice(
            self.data,
            [0, batch_index * self.time_steps],
            [self.batch_size, (batch_index + 1) * self.time_steps])
        x_batch.set_shape(self.batch_shape)

        y_batch = tf.strided_slice(
            self.data,
            [0, batch_index * self.time_steps + 1],
            [self.batch_size, (batch_index + 1) * self.time_steps + 1])
        y_batch.set_shape(self.batch_shape)

        return x_batch, y_batch
