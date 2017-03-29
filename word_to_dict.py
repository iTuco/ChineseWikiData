try:
    import lzma
except ImportError:
    from backports import lzma
import collections
import multiprocessing
import os

import jieba

SUB_PROCESS = 4


def tokenizer(filename):
    word_counter = collections.defaultdict(int)
    with open(filename) as f:
        for line in f:
            for word in jieba.tokenize(line.decode('utf-8')):
                word_counter[word[0]] += 1
    return word_counter


def main(path, outfile):
    pool = multiprocessing.Pool(SUB_PROCESS)
    _, _, filenames = next(os.walk(path))

    for i, filename in enumerate(filenames):
        filenames[i] = os.path.join(path, filename)

    word_dict = None

    for wd in pool.imap_unordered(tokenizer, filenames):
        if word_dict is None:
            word_dict = wd
            break
        else:
            for key in wd:
                word_dict[key] += wd[key]

    word_kv = []
    for key in word_dict:
        word_kv.append((key, word_dict[key]))

    word_kv.sort(key=lambda x: x[1], reverse=True)

    with open(outfile, 'w') as of:
        for w, c in word_kv:
            print >> of, w, c,

if __name__ == '__main__':
    main("./data", 'word_dict')
