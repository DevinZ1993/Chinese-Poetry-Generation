#! /usr/bin/env python
# -*- coding:utf-8 -*-

from data_util import *


if __name__ == '__main__':
    keywords,_ = get_keywords()
    idx = 0
    for keyword in keywords:
        idx += 1
        if idx > 100:
            break
        if 0 == idx%5:
            uprintln(keyword[0])
        else:
            uprint(keyword[0]+u'\t')
    print
    print "Total keywords: %d" %len(keywords)
