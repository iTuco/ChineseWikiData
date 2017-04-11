# *-* coding=utf-8 *-*
import os

import jieba
import regex as re
import multiprocessing

pattern = re.compile(r'([\p{IsHan}]+)', re.UNICODE)
end_of_sentence = [u"。", u"？", u"！"]


def cut_line(line):
    line = line.decode('utf-8')
    words = list(jieba.cut(line))
    return map(
        lambda x: x.encode('utf-8') if x not in end_of_sentence else "\n",
        filter(lambda w: w in end_of_sentence or pattern.match(
            w) is not None, words))


def main(path, outfile):
    _, _, filenames = next(os.walk(path))
    pool = multiprocessing.Pool(3)
    with open(outfile, 'w') as f:
        for filename in filenames:
            with open(os.path.join(path, filename)) as fin:
                lines = (line for line in fin)
                for words in pool.imap(cut_line, lines, chunksize=400):
                    for w in words:
                        f.write(w)
                        f.write(' ')
                f.write("\n")


if __name__ == '__main__':
    main("./data", "./all_file")
