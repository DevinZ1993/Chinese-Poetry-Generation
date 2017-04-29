#! /usr/bin/env python
# -*- coding:utf-8 -*-

from data_utils import *
from plan import Planner
from generate import Generator
import sys


reload(sys)
sys.setdefaultencoding('utf8')


if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        line = raw_input('Input Text:\t').decode('utf-8').strip()
        if line.lower() == 'quit' or line.lower() == 'exit':
            break
        elif len(line) > 0:
            keywords = planner.plan(line)
            print "Keywords:\t",
            for word in keywords:
                print word,
            print '\n'
            print "Poem Generated:\n"
            sentences = generator.generate(keywords)
            print '\t'+sentences[0]+u'，'+sentences[1]+u'。'
            print '\t'+sentences[2]+u'，'+sentences[3]+u'。'
            print

