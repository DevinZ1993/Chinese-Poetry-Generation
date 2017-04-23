#! /usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import sys
import os
import json
import random
import numpy as np


raw_dir = 'raw'
data_dir = 'data'
save_dir = 'save'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def uprint(x):
    print repr(x).decode('unicode-escape'),

def uprintln(x):
    print repr(x).decode('unicode-escape')

def is_CN_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def split_sentences(line):
    sentences = []
    i = 0
    for j in range(len(line)+1):
        if j == len(line) or line[j] in [u'，', u'。', u'！', u'？', u'、']:
            if i < j:
                sentence = u''.join(filter(is_CN_char, line[i:j]))
                sentences.append(sentence)
            i = j+1
    return sentences

