import paddle.v2 as paddle
import sys
import jieba
import os
import gzip

CPU_NUM = 4


def reader(window_size, word_dict, filename):
    words = []
    with open(filename) as f:
        for line in f:
            for word in jieba.tokenize(line.decode('utf-8')):
                word_id = word_dict.get(word[0], None)
                if word_id is not None:
                    words.append(word_id)
                    if len(words) == window_size:
                        yield words[:window_size / 2] + words[
                                                        -window_size / 2:] + \
                              [words[window_size / 2 + 1]]
                        words.pop(0)


def reader_creator(window_size, word_dict, path):
    def __impl__():
        _, _, filenames = next(os.walk(path))
        for each_filename in filenames:
            for item in reader(window_size, word_dict,
                               os.path.join(path, each_filename)):
                yield item

    return __impl__


def main(window_size=5):
    assert window_size % 2 == 1
    paddle.init(use_gpu=False, trainer_count=CPU_NUM)
    word_dict = dict()
    with open('word_dict') as f:
        for word_id, line in enumerate(f):
            w, wc = line.split()
            word_dict[w.decode('utf-8')] = word_id

    word_left = []
    word_right = []

    for i in xrange(window_size / 2):
        word_left.append(paddle.layer.data(name='word_left_%d' % i,
                                           type=paddle.data_type.integer_value(
                                               len(word_dict))))
        word_right.append(paddle.layer.data(name='word_right_%d' % i,
                                            type=paddle.data_type.integer_value(
                                                len(word_dict))))

    embs = []
    for w in word_left + word_right:
        embs.append(
            paddle.layer.embedding(input=w, size=16, param_attr=
            paddle.attr.Param(name='emb', sparse_update=True)))

    contextemb = paddle.layer.concat(input=embs)

    hidden1 = paddle.layer.fc(input=contextemb,
                              size=32,
                              act=paddle.activation.Sigmoid())
    predictword = paddle.layer.fc(input=hidden1,
                                  size=len(word_dict),
                                  bias_attr=paddle.attr.Param(learning_rate=2),
                                  act=paddle.activation.Softmax())

    cost = paddle.layer.classification_cost(input=predictword,
                                            label=paddle.layer.data(
                                                name='mid_word',
                                                type=paddle.data_type.integer_value(
                                                    len(word_dict))))
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.AdaGrad(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adam_optimizer)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            sys.stdout.write('.')
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            if event.batch_id % 1000 == 0:
                with gzip.open("model_%d_%d.tar.gz" % (event.pass_id,
                                                       event.batch_id),
                               'w') as f:
                    parameters.to_tar(f)

        if isinstance(event, paddle.event.EndPass):
            print "Pass %d" % event.pass_id
            with gzip.open("model_%d.tar.gz" % event.pass_id, 'w') as f:
                parameters.to_tar(f)

    trainer.train(
        paddle.batch(
            paddle.reader.buffered(
                reader_creator(window_size=window_size, word_dict=word_dict,
                               path="./data"), 16 * CPU_NUM * 1000),
            16 * CPU_NUM),
        num_passes=1,
        event_handler=event_handler)


if __name__ == '__main__':
    main()
