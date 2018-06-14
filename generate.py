#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from common import *
import os


def train_generator():
    print("Training RNN-based generator ...")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

