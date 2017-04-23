#! /usr/bin/env python
# -*- coding:utf-8 -*-

from utils import *
from data_utils import *
from generate import Generator
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


if __name__ == '__main__':
    generator = Generator()
    with tf.Session() as sess:
        generator.train(sess)

