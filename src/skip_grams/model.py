import tensorflow as tf

from common.evaluation import Evaluator
from common.model import WordToVecModel
from .batcher import SkipGramBatcher


class SkipGramModel(WordToVecModel):

    def __init__(self, batcher: 'SkipGramBatcher', evaluator: 'Evaluator', embedding_dim: 'int'):
        super().__init__(batcher, evaluator, embedding_dim)

        w_to_v_shape = (embedding_dim, batcher.x_shape[0])
        w_out_shape = (batcher.y_shape[0], embedding_dim)
        b_out_shape = (batcher.y_shape[0], 1)

        self._w_to_v = tf.get_variable('w_to_v', shape=w_to_v_shape)
        self._w_out = tf.get_variable('w_out', shape=w_out_shape)
        self._b_out = tf.get_variable('b_out', shape=b_out_shape)

        self._target = tf.placeholder(dtype=tf.float32, shape=batcher.x_shape)
        self._context = tf.placeholder(dtype=tf.float32, shape=batcher.y_shape)

    def _get_train_op(self, learning_rate: 'float') -> 'tf.Tensor':
        l1_a = tf.matmul(self._w_to_v, self._target)
        l2_z = tf.matmul(self._w_out, l1_a) + self._b_out
        l2_a = tf.nn.softmax(l2_z, axis=0)
        loss = tf.losses.log_loss(self._context, l2_a)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(loss=loss)
