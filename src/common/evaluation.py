import tensorflow as tf

from .batcher import SimilarityBatcher


class Evaluator:

    def evaluate(self, w_to_v: 'tf.Variable', sess: 'tf.Session') -> 'float':
        raise NotImplementedError()


class SimilarityEvaluator(Evaluator):

    def __init__(self, batcher: 'SimilarityBatcher'):
        self._batcher = batcher
        self._x1s_ph = tf.placeholder(dtype=tf.float32, shape=batcher.xs_shape)
        self._x2s_ph = tf.placeholder(dtype=tf.float32, shape=batcher.xs_shape)
        self._ys_ph = tf.placeholder(dtype=tf.float32, shape=batcher.ys_shape)

    @staticmethod
    def get_vecs_lengths(vecs: 'tf.Tensor') -> 'tf.Tensor':
        return tf.transpose(tf.sqrt(tf.reduce_sum(tf.multiply(vecs, vecs), axis=1)))

    def _get_evaluate_op(self, w_to_v: 'tf.Variable') -> 'tf.Tensor':
        """
        Batched version of cosine similarity algorithm.
        :param w_to_v:
        :return: summed up cosine similarities: float
        """
        x1_vecs = tf.matmul(w_to_v, self._x1s_ph)
        x2_vecs = tf.matmul(w_to_v, self._x2s_ph)
        dot_products = tf.transpose(tf.reduce_sum(tf.multiply(x1_vecs, x2_vecs), axis=1))
        x1_lens, x2_lens = self.get_vecs_lengths(x1_vecs), self.get_vecs_lengths(x2_vecs)
        cos_alphas = tf.divide(dot_products, tf.multiply(x1_lens, x2_lens))
        return tf.reduce_sum(tf.acos(cos_alphas))

    def evaluate(self, w_to_v: 'tf.Variable', sess: 'tf.Session') -> 'float':
        loss = 0.0
        evaluate_op = self._get_evaluate_op(w_to_v)
        for x1s, x2s, ys in self._batcher:
            feed_dict = {self._x1s_ph: x1s, self._x2s_ph: x2s, self._ys_ph: ys}
            loss += sess.run(evaluate_op, feed_dict=feed_dict)
        print('')
        return loss
