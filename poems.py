#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char_dict import CharDict
from paths import raw_dir, poems_path, check_uptodate
from random import shuffle
from singleton import Singleton
from utils import split_sentences
import os

_corpus_list = ['qts_tab.txt']


def _gen_poems():
    print("Parsing poems ...")
    char_dict = CharDict()
    with open(poems_path, 'w') as fout:
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
        if not check_uptodate(poems_path):
            _gen_poems()
        self.poems = []
        with open(poems_path, 'r') as fin:
            for line in fin.readlines():
                self.poems.append(line.strip().split())

    def __getitem__(self, index):
        if index < 0 or index >= len(self.poems):
            return None
        return self.poems[index]

    def __len__(self):
        return len(self.poems)

    def __iter__(self):
        return iter(self.poems)

    def shuffle(self):
        shuffle(self.poems)


# For testing purpose.
if __name__ == '__main__':
    poems = Poems()
    for i in range(10):
        print(' '.join(poems[i]))

