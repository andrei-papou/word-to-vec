import random
from collections import Iterable
from typing import Tuple
import numpy as np
import attr
from common.utils import to_one_hot


@attr.s(frozen=True)
class Window:
    left = attr.ib(type=int)
    right = attr.ib(type=int)

    @classmethod
    def symmetric(cls, size: int) -> 'Window':
        return cls(left=size, right=size)

    @property
    def size(self):
        return self.left + self.right


class SimilarityBatcher(Iterable):

    def __init__(self, x1s: 'np.array', x2s: 'np.array', ys: 'np.array', batch_size: 'int', vocab_size: 'int'):
        self._x1s = x1s
        self._x2s = x2s
        self._ys = ys
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._index = 0
        self._data_len = x1s.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index + self._batch_size > self._data_len:
            self._index = 0
            raise StopIteration()
        slc = slice(self._index, self._index + self._batch_size)
        x1s = to_one_hot(self._x1s[slc], self._vocab_size).T
        x2s = to_one_hot(self._x2s[slc], self._vocab_size).T
        ys = self._ys[slc].reshape(self._batch_size, 1)
        self._index += self._batch_size
        return x1s, x2s, ys

    @property
    def xs_shape(self) -> ('int', 'int'):
        return self._vocab_size, self._batch_size

    @property
    def ys_shape(self) -> ('int', 'int'):
        return self._batch_size, 1


class WordToVecBatcher:

    def __init__(self, data: 'np.array', window: 'Window', batch_size: 'int', vocab_size: 'int'):
        self._data = data
        self._window = window
        self._batch_size = batch_size
        self._vocab_size = vocab_size

        self._epoch = 0
        self._reset_targets()

    def get_batch(self) -> 'Tuple[np.array, np.array]':
        raise NotImplementedError()

    @property
    def epoch(self) -> 'int':
        return self._epoch

    @property
    def window_size(self) -> 'int':
        return self._window.size

    @property
    def batch_size(self) -> 'int':
        return self._batch_size

    @property
    def x_shape(self) -> 'Tuple':
        raise NotImplementedError()

    @property
    def y_shape(self) -> 'Tuple':
        raise NotImplementedError()

    @property
    def left_targets(self) -> 'int':
        return len(self._targets)

    @property
    def num_targets(self) -> 'int':
        return self._data.shape[0] - self._window.size

    def _reset_targets(self):
        lst = list(range(self._data.shape[0] - self._window.size))
        random.shuffle(lst)
        self._targets = lst

    def _get_target(self) -> 'int':
        if not self._targets:
            self._reset_targets()
            self._epoch += 1
        return self._targets.pop()

    def _get_one_hot_by_index(self, index: 'int') -> 'np.array':
        return to_one_hot(np.array([self._data[index]]), self._vocab_size)
