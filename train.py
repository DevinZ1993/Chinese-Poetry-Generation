#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from utils import *
from generate import Generator
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
        generator = Generator()
        generator.train()
    print("All training is done!")

