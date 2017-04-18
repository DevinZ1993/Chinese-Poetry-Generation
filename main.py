#! /usr/bin/env python
# -*- coding:utf-8 -*-

from data_util import *
from plan import Planner


if __name__ == '__main__':
    planner = Planner()
    while True:
        line = raw_input('Text:\t').decode('utf-8').strip()
        if line.lower() == 'quit' or line.lower() == 'exit':
            break
        elif len(line) > 0:
            keywords = planner.plan(line)
            print "Words:\t",
            for word in keywords:
                print repr(word).decode('unicode-escape'),
            print '\n'

