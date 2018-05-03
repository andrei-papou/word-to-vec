import tensorflow as tf

from common.evaluation import Evaluator
from common.model import WordToVecModel
from .batcher import CBOWBatcher


class CBOWModel(WordToVecModel):

    def __init__(self, batcher: 'CBOWBatcher', evaluator: 'Evaluator', embedding_dim: 'int'):
        super().__init__(batcher, evaluator, embedding_dim)

        w_to_v_shape = (embedding_dim, batcher.x_shape[1])
        w_out_shape = (batcher.y_shape[0], embedding_dim * batcher.window_size)
        b_out_shape = (batcher.y_shape[0], 1)

        self._w_to_v = tf.get_variable('w_to_v', shape=w_to_v_shape)
        self._w_out = tf.get_variable('w_out', shape=w_out_shape)
        self._b_out = tf.get_variable('b_out', shape=b_out_shape)

        self._context = tf.placeholder(dtype=tf.float32, shape=batcher.x_shape)
        self._target = tf.placeholder(dtype=tf.float32, shape=batcher.y_shape)

    def _get_train_op(self, learning_rate: 'float') -> 'tf.Tensor':
        # TODO: apply _w_to_v window size times in loop and then stack the result
        ctx_size = self._context.shape[0]
        l1_a_parts = [tf.matmul(self._w_to_v, self._context[c, :, :]) for c in range(ctx_size)]
        l1_a = tf.concat(l1_a_parts, axis=1)
        l1_a = tf.reshape(l1_a, shape=(self._embedding_dim * self._batcher.window_size, self._batcher.batch_size))
        l2_z = tf.matmul(self._w_out, l1_a) + self._b_out
        target_prediction = tf.nn.softmax(l2_z, axis=0)
        loss = tf.losses.log_loss(self._target, target_prediction)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(loss=loss)
