#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os

root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data')
raw_dir = os.path.join(data_dir, 'raw')
save_dir = os.path.join(root_dir, 'save')

plan_data_path = os.path.join(data_dir, 'plan_data.txt')
gen_data_path = os.path.join(data_dir, 'gen_data.txt')

plan_model_path = os.path.join(save_dir, 'plan_model.bin')
gen_model_path = os.path.join(save_dir, 'gen_model.bin')

# Test if a char is a Chinese character.
def is_cn_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

# Test if a sentence is made of Chinese charaters.
def is_cn_sentence(sentence):
    for ch in sentence:
        if not is_cn_char(ch):
            return False
    return True

# Split a piece of text into a list of sentences.
def split_sentences(text):
    sentences = []
    i = 0
    for j in range(len(text) + 1):
        if j == len(text) or \
                text[j] in [u'，', u'。', u'！', u'？', u'、', u'\n']:
            if i < j:
                sentence = u''.join(filter(is_cn_char, text[i:j]))
                sentences.append(sentence)
            i = j + 1
    return sentences

NUM_OF_SENTENCES = 4
CHAR_VEC_DIM = 512
