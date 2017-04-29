#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter
from vocab import get_vocab, VOCAB_SIZE
from quatrains import get_quatrains
from gensim import models
from numpy.random import uniform

_w2v_path = os.path.join(data_dir, 'word2vec.npy')

def _gen_embedding(ndim):
    print "Generating %d-dim word embedding ..." %ndim
    int2ch, ch2int = get_vocab()
    ch_lists = []
    quatrains = get_quatrains()
    for idx, poem in enumerate(quatrains):
        for sentence in poem['sentences']:
            ch_lists.append(filter(lambda ch: ch in ch2int, sentence))
        if 0 == (idx+1)%10000:
            print "[Word2Vec] %d/%d poems have been processed." %(idx+1, len(quatrains))
    print "Hold on. This may take some time ..."
    model = models.Word2Vec(ch_lists, size = ndim, min_count = 5)
    embedding = uniform(-1.0, 1.0, [VOCAB_SIZE, ndim])
    for idx, ch in enumerate(int2ch):
        if ch in model.wv:
            embedding[idx,:] = model.wv[ch]
    np.save(_w2v_path, embedding)
    print "Word embedding is saved."

def get_word_embedding(ndim):
    if not os.path.exists(_w2v_path):
        _gen_embedding(ndim)
    return np.load(_w2v_path)


if __name__ == '__main__':
    embedding = get_word_embedding(128)
    print "Size of embedding: (%d, %d)" %embedding.shape


