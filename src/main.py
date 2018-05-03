from common.reader import read_train_data, Vocab, read_evaluation_data
from common.batcher import SimilarityBatcher, Window
from common.evaluation import SimilarityEvaluator

from cbow.batcher import CBOWBatcher
from cbow.model import CBOWModel

from skip_grams.batcher import SkipGramBatcher
from skip_grams.model import SkipGramModel


if __name__ == '__main__':
    vocab = Vocab.build('ptb.train.txt')
    x1s, x2s, ys = read_evaluation_data('similarity1.csv', vocab=vocab)
    ev_batcher = SimilarityBatcher(x1s=x1s, x2s=x2s, ys=ys, batch_size=10, vocab_size=vocab.size)
    evaluator = SimilarityEvaluator(batcher=ev_batcher)

    batcher = SkipGramBatcher(
        data=read_train_data('ptb.train.txt', vocab=vocab),
        window=Window.symmetric(3),
        batch_size=50,
        vocab_size=vocab.size)
    model = SkipGramModel(batcher=batcher, evaluator=evaluator, embedding_dim=64)
    model.train(epochs=100, learning_rate=0.05)
