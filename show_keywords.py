#! /usr/bin/env python

from data_util import *


if __name__ == '__main__':
    corpus = Corpus()
    keywords = corpus.get_keywords()
    idx = 0
    for keyword in keywords:
        idx += 1
        if 10 == idx:
            idx = 0
            uprint(keyword)
        else:
            print repr(keyword).decode('unicode-escape')+'\t',
    print
    print "Total keywords: %d" %len(keywords)
