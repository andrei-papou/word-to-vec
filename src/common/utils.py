import numpy as np


def to_one_hot(a: 'np.array', data_len: 'int') -> 'np.array':
    b = np.zeros((a.size, data_len))
    b[np.arange(a.size), a] = 1
    return b
