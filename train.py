#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from common import *
from generate import train_generator
from gensim import models
from plan import train_planner
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Chinese poem generation.')
    parser.add_argument('-p', dest = 'planner', default = False, 
            action = 'store_true', help = 'train planning model')
    parser.add_argument('-g', dest = 'generator', default = False, 
            action = 'store_true', help = 'train generation model')
    parser.add_argument('-a', dest = 'all', default = False,
            action = 'store_true', help = 'train both models')
    args = parser.parse_args()
    if args.all or args.planner:
        train_planner()
    if args.all or args.generator:
        train_generator()
    print("All training is done!")

