#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import common
import generator

if __name__ == '__main__':
    common.global_init()
    gen = generator.Generator()
    while True:
        head = input('>>> ')
        poem = gen.generate(head)
        for sentence in poem:
            print(sentence)
        print()
