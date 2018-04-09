from threading import Thread

import tensorflow as tf

from reader import DataReader


def thread_func(coord: tf.train.Coordinator):
    i = 1
    while not coord.should_stop():
        print(i)
        i += 1
        if i == 5:
            break


if __name__ == '__main__':
    data_reader = DataReader()

    coord = tf.train.Coordinator()
    threads = [Thread(target=thread_func, args=(coord,)) for _ in range(3)]

    for t in threads:
        t.start()

    coord.join(threads)

    # train_subset = data_reader.train[100:105]
    # valid_subset = data_reader.valid[100:105]
    # test_subset = data_reader.test[100:105]
    #
    # print(train_subset)
    # print(valid_subset)
    # print(test_subset)
    # print(data_reader.vocab_size)
