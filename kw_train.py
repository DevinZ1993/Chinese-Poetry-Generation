#! /usr/bin/env python
#-*- coding:utf-8 -*-

from plan import Planner
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


if __name__ == '__main__':
    planner = Planner()
    with tf.Session() as sess:
        planner.train(sess)

