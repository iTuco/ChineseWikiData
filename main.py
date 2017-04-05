# coding=utf-8
import paddle.v2 as paddle
import sys
import jieba
import os
import gzip
import regex as re

CPU_NUM = 4
EMB_SIZE = 32
HIDDEN_SIZE = 128
WORD_DICT_LIMIT = 200000


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


pattern = re.compile(r'([\p{IsHan}\p{IsBopo}\p{IsHira}\p{IsKatakana}]+)',
                     re.UNICODE)


def main(window_size=5):
    assert window_size % 2 == 1
    paddle.init(use_gpu=False, trainer_count=CPU_NUM)
    word_dict = dict()
    with open('word_dict') as f:
        word_id = 0
        for line in f:
            if word_id > WORD_DICT_LIMIT: break
            w, wc = line.split()
            w = w.decode('utf-8')
            if pattern.match(w) is None:
                continue
            word_dict[w] = word_id
            word_id += 1

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
            paddle.layer.embedding(input=w, size=EMB_SIZE, param_attr=
            paddle.attr.Param(name='emb', sparse_update=True)))

    contextemb = paddle.layer.concat(input=embs)

    hidden1 = paddle.layer.fc(input=contextemb,
                              size=HIDDEN_SIZE,
                              act=paddle.activation.Sigmoid())

    cost = paddle.layer.hsigmoid(input=hidden1,
                                 label=paddle.layer.data(
                                     name='mid_word',
                                     type=paddle.data_type.integer_value(
                                         len(word_dict))),
                                 num_classes=WORD_DICT_LIMIT)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.AdaGrad(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost, parameters, adam_optimizer)

    counter = [0]
    total_cost = [0.0]

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            total_cost[0] += event.cost
            counter[0] += 1

            sys.stdout.write('.')
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, AvgCost %f" % (
                    event.pass_id, event.batch_id, total_cost[0] / counter[0])
            if event.batch_id % 10000 == 0:
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
            32 * CPU_NUM),
        num_passes=50,
        event_handler=event_handler)


if __name__ == '__main__':
    main()
