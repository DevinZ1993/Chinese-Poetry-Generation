#! /usr/bin/env python
# -*- coding:utf-8 -*-

from data_util import *
from generate import Generator
import tensorflow as tf


if __name__ == '__main__':
    generator = Generator()
    with tf.Session() as sess:
        generator.train(sess, int(sys.argv[1]), int(sys.argv[2]))

