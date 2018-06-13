#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from common import *
from singleton import Singleton
from char_dict import CharDict

_poem_path = os.path.join(data_dir, 'poem.txt')

_corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qtais_tab.txt',
        'yuan.all', 'ming.all', 'qing.all']


def _gen_poems():
    print("Parsing poems ...")
    char_dict = CharDict()
    with open(_poem_path, 'w') as fout:
        for corpus in _corpus_list:
            with open(os.path.join(raw_dir, corpus), 'r') as fin:
                for line in fin.readlines()[1 : ]:
                    sentences = split_sentences(line.strip().split()[-1])
                    all_char_in_dict = True
                    for sentence in sentences:
                        for ch in sentence:
                            if char_dict.char2int(ch) < 0:
                                all_char_in_dict = False
                                break
                        if not all_char_in_dict:
                            break
                    if all_char_in_dict:
                        fout.write(' '.join(sentences) + '\n')
            print("Finished parsing %s." % corpus)


class Poems(Singleton):

    def __init__(self):
        if not os.path.exists(_poem_path):
            _gen_poems()
        self.poems = []
        with open(_poem_path, 'r') as fin:
            for line in fin.readlines():
                self.poems.append(line.strip().split())

    def __getitem__(self, index):
        if index < 0 or index >= len(self.poems):
            return None
        return self.poems[index]


# For testing purpose.
if __name__ == '__main__':
    poems = Poems()
    for i in range(10):
        print(' '.join(poems[i]))

