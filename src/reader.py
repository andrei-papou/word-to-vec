import os
from typing import List

from tensorflow import gfile

import constants


def _read_words(filename: str) -> List[str]:
    with gfile.GFile(filename)as f:
        return f.read().replace('\n', constants.EOS).split()


def get_data(data_path: str):
    train_path = os.path.join(data_path, constants.TRAIN_FILE_NAME)
    test_path = os.path.join(data_path, constants.TEST_FILE_NAME)
    valid_path = os.path.join(data_path, constants.VALID_FILE_NAME)

    train_words = _read_words(train_path)
    test_words = _read_words(test_path)

    print('Train ==> length: {}, examples: {}'.format(len(train_words), train_words[:3]))
    print('Test  ==> length: {}, examples: {}'.format(len(test_words), test_words[:3]))
