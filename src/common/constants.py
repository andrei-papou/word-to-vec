from enum import Enum


DATA_PATH = '../.data'


class DataSetType(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


TRAIN_FILE_NAME = 'ptb.train.txt'
TEST_FILE_NAME = 'ptb.test.txt'
VALID_FILE_NAME = 'ptb.valid.txt'

EOS = '<eos>'
UNK = '<unk>'
