import tensorflow as tf

from reader import DataReader, BatchProducer


def thread_func(coord: tf.train.Coordinator):
    i = 1
    while not coord.should_stop():
        print(i)
        i += 1
        if i == 5:
            break


if __name__ == '__main__':
    data_reader = DataReader()
    batch_producer = BatchProducer(raw_data=data_reader.train, batch_size=50, time_steps=10)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners()

        print(sess.run(batch_producer.build_batch()))
        print(sess.run(batch_producer.build_batch()))

    # train_subset = data_reader.train[100:105]
    # valid_subset = data_reader.valid[100:105]
    # test_subset = data_reader.test[100:105]
    #
    # print(train_subset)
    # print(valid_subset)
    # print(test_subset)
    # print(data_reader.vocab_size)
