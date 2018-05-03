from typing import List, Tuple

import numpy as np

from common.batcher import WordToVecBatcher


class CBOWBatcher(WordToVecBatcher):

    @property
    def x_shape(self) -> 'Tuple[int, int, int]':
        return self._window.size, self._vocab_size, self._batch_size

    @property
    def y_shape(self) -> 'Tuple[int, int]':
        return self._vocab_size, self._batch_size

    def get_batch(self) -> 'Tuple[np.array, np.array]':
        """
        Let WL - left window size, WR - right window size, VS - vocab size, BS - batch size.
        Last dimension is reserved for batches.
        X shape: (WL + WR) x VS x BS
        Y shape: VS x BS
        :return: X, Y
        """
        contexts, targets = [], []
        for _ in range(self._batch_size):
            c, t = self._get_bag()
            contexts.append(c)
            targets.append(t)
        return np.concatenate(contexts, axis=2), np.concatenate(targets, axis=1)

    def _get_bag(self) -> ('List[np.array]', 'np.array'):
        """
        Let WL - left window size, WR - right window size, VS - vocab size.
        Last dimension is reserved for batches.
        X shape: (WL + WR) x VS x 1
        Y shape: VS x 1
        :return: X, Y
        """
        target = self._get_target()
        left_context = [
            self._get_one_hot_by_index(target - i).reshape(1, self._vocab_size, 1)
            for i in reversed(range(1, self._window.left + 1))
        ]
        right_context = [
            self._get_one_hot_by_index(target + i).reshape(1, self._vocab_size, 1)
            for i in range(1, self._window.right + 1)
        ]
        target = self._get_one_hot_by_index(target).reshape(self._vocab_size, 1)
        return np.concatenate(left_context + right_context, axis=0), target
