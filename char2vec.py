#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char_dict import CharDict
from gensim import models
from numpy.random import uniform
from paths import char2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import CHAR_VEC_DIM
import numpy as np
import os


def _gen_char2vec():
    print("Generating char2vec model ...")
    char_dict = CharDict()
    poems = Poems()
    model = models.Word2Vec(poems, size = CHAR_VEC_DIM, min_count = 5)
    embedding = uniform(-1.0, 1.0, [len(char_dict), CHAR_VEC_DIM])
    for i, ch in enumerate(char_dict):
        if ch in model.wv:
            embedding[i, :] = model.wv[ch]
    np.save(char2vec_path, embedding)


class Char2Vec(Singleton):

    def __init__(self):
        if not check_uptodate(char2vec_path):
            _gen_char2vec()
        self.embedding = np.load(char2vec_path)
        self.char_dict = CharDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.embedding[self.char_dict.char2int(ch)]

    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, CHAR_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    char2vec = Char2Vec()

