#! /usr/bin/env python
#-*- coding: utf-8 -*-

from utils import *
from segment import Segmenter
from quatrains import get_quatrains

_wc_path = os.path.join(data_dir, 'word_cnts.json')


def _gen_word_cnts():
    counters = dict()
    segmenter = Segmenter()
    quatrains = get_quatrains()
    for idx, poem in enumerate(quatrains):
        for sentence in poem['sentences']:
            segs = segmenter.segment(sentence)
            for seg in segs:
                counters[seg] = counters[seg]+1 if seg in counters else 1
        if 0 == (idx+1)%10000:
            print "[Word Count] %d/%d quatrains has been processed." %(idx+1, len(quatrains))
    with codecs.open(_wc_path, 'w', 'utf-8') as fout:
        json.dump(counters, fout)

def get_word_cnts():
    if not os.path.exists(_wc_path):
        _gen_word_cnts()
    with codecs.open(_wc_path, 'r', 'utf-8') as fin:
        return json.load(fin)

def _min_word_cnt(cnts, poem, segmenter):
    min_cnt = (1<<31)-1
    for sentence in poem['sentences']:
        segs = segmenter.segment(sentence)
        for seg in segs:
            min_cnt = min(min_cnt, cnts[seg])
    return min_cnt

def get_pop_quatrains(num = 10000):
    cnts = get_word_cnts()
    segmenter = Segmenter()
    quatrains = get_quatrains()
    min_word_cnts = [_min_word_cnt(cnts, quatrain, segmenter) \
            for i, quatrain in enumerate(quatrains)]
    indexes = sorted(range(len(quatrains)), key = lambda i: -min_word_cnts[i])
    return [quatrains[index] for index in indexes[:min(num, len(indexes))]]

if __name__ == '__main__':
    cnts = get_word_cnts()
    words = sorted([word for word in cnts], key = lambda w: -cnts[w])
    uprintln(words[:20])

