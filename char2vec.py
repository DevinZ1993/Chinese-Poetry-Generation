#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from common import *
from singleton import Singleton
from char_dict import CharDict
from poems import Poems
import numpy as np
from numpy.random import uniform
from gensim import models

_char2vec_path = os.path.join(data_dir, 'char2vec.npy')

def _gen_char2vec():
    print("Generating char2vec model ...")
    char_dict = CharDict()
    poems = Poems()
    model = models.Word2Vec(poems, size = CHAR_VEC_DIM, min_count = 5)
    embedding = uniform(-1.0, 1.0, [len(char_dict), CHAR_VEC_DIM])
    for i, ch in enumerate(char_dict):
        if ch in model.wv:
            embedding[i, :] = model.wv[ch]
    np.save(_char2vec_path, embedding)


class Char2Vec(Singleton):

    def __init__(self):
        if not os.path.exists(_char2vec_path):
            _gen_char2vec()
        self.embedding = np.load(_char2vec_path)
        self.char_dict = CharDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.char2vec[self.char2int(ch)]


# For testing purpose.
if __name__ == '__main__':
    char2vec = Char2Vec()

