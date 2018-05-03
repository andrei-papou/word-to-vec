import tensorflow as tf

from .evaluation import Evaluator
from .batcher import WordToVecBatcher


class WordToVecModel:

    def __init__(self, batcher: 'WordToVecBatcher', evaluator: 'Evaluator', embedding_dim: 'int'):
        self._batcher = batcher
        self._evaluator = evaluator
        self._embedding_dim = embedding_dim

        self._context = None
        self._target = None
        self._w_to_v = None

    def _get_train_op(self, learning_rate: 'float') -> 'tf.Tensor':
        raise NotImplementedError()

    def train(self, epochs: 'int', learning_rate: 'float'):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            while self._batcher.epoch < epochs:
                context, target = self._batcher.get_batch()
                train_op = self._get_train_op(learning_rate)
                sess.run(train_op, feed_dict={self._context: context, self._target: target})

                print('Epoch: {epoch}, targets: {targets} / {total}, loss: {loss}'.format(
                    epoch=self._batcher.epoch,
                    targets=self._batcher.num_targets - self._batcher.left_targets,
                    total=self._batcher.num_targets,
                    loss=self._evaluator.evaluate(self._w_to_v, sess)))
