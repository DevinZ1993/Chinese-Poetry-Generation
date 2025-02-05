#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import common
import generator
import vocab

if __name__ == '__main__':
    common.global_init()
    vocab_dict = vocab.Vocab(vocab.EMBEDDING_DIM)
    gen = generator.Generator(vocab_dict)
    while True:
        head = input('>>> ')
        poem = gen.generate(head)
        for sentence in poem:
            print(sentence)
        print()
