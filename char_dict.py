#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import raw_dir, char_dict_path, check_uptodate
from singleton import Singleton
from utils import is_cn_char
import os


MAX_DICT_SIZE = 6000

_corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
        'yuan.all', 'ming.all', 'qing.all']


def start_of_sentence():
    return '^'

def end_of_sentence():
    return '$'

def _gen_char_dict():
    print("Generating dictionary from corpus ...")
    
    # Count char frequencies.
    char_cnts = dict()
    for corpus in _corpus_list:
        with open(os.path.join(raw_dir, corpus), 'r') as fin:
            for ch in filter(is_cn_char, fin.read()):
                if ch not in char_cnts:
                    char_cnts[ch] = 1
                else:
                    char_cnts[ch] += 1
    
    # Sort in decreasing order of frequency.
    cnt2char = sorted(char_cnts.items(), key = lambda x: -x[1])

    # Store most popular chars into the file.
    with open(char_dict_path, 'w') as fout:
        for i in range(min(MAX_DICT_SIZE - 2, len(cnt2char))):
            fout.write(cnt2char[i][0])


class CharDict(Singleton):

    def __init__(self):
        if not check_uptodate(char_dict_path):
            _gen_char_dict()
        self._int2char = []
        self._char2int = dict()
        # Add start-of-sentence symbol.
        self._int2char.append(start_of_sentence())
        self._char2int[start_of_sentence()] = 0
        with open(char_dict_path, 'r') as fin:
            idx = 1
            for ch in fin.read():
                self._int2char.append(ch)
                self._char2int[ch] = idx
                idx += 1
        # Add end-of-sentence symbol.
        self._int2char.append(end_of_sentence())
        self._char2int[end_of_sentence()] = len(self._int2char) - 1

    def char2int(self, ch):
        if ch not in self._char2int:
            return -1
        return self._char2int[ch]

    def int2char(self, idx):
        return self._int2char[idx]

    def __len__(self):
        return len(self._int2char)

    def __iter__(self):
        return iter(self._int2char)

    def __contains__(self, ch):
        return ch in self._char2int


# For testing purpose.
if __name__ == '__main__':
    char_dict = CharDict()
    for i in range(10):
        ch = char_dict.int2char(i)
        print(ch)
        assert i == char_dict.char2int(ch)

