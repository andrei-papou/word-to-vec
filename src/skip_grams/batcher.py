import random
from typing import Tuple
import numpy as np

from common.batcher import WordToVecBatcher


class SkipGramBatcher(WordToVecBatcher):

    @property
    def x_shape(self) -> 'Tuple[int, int]':
        return self._vocab_size, self._batch_size

    @property
    def y_shape(self) -> 'Tuple[int, int]':
        return self._vocab_size, self._batch_size

    def get_batch(self) -> 'Tuple[np.array, np.array]':
        contexts, targets = [], []
        for _ in range(self.batch_size):
            ti = self._get_target()
            left_is = [ti - i for i in range(1, self._window.left)]
            right_is = [ti - i for i in range(1, self._window.right)]
            ci = random.choice(left_is + right_is)
            contexts.append(self._get_one_hot_by_index(ci).T)
            targets.append(self._get_one_hot_by_index(ti).T)
        return np.concatenate(targets, axis=1), np.concatenate(contexts, axis=1)
