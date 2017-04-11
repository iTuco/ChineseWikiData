import word2vec
#
# word2vec.word2phrase(train="./all_file", output="./all_file_phrase",
#                     verbose=True)
word2vec.word2vec(train="./all_file",
                  output="./w2v.txt",
                  size=128,
                  window=9,
                  verbose=True,
                  binary=0)