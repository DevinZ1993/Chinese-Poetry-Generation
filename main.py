#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from generate import Generator
from plan import Planner


if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        hints = input("Type in hints >> ")
        keywords = planner.plan(hints)
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords)
        print("Poem generated:")
        for sentence in poem:
            print(sentence)

