#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from generate import Generator
from multiprocessing import Process
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def _rand_train(batch_num):
    generator = Generator()
    with tf.Session() as sess:
        generator.rand_train(sess, batch_num)


if __name__ == '__main__':
    batch_num = 128
    batch_cnt = 0
    while True:
        proc = Process(target = _rand_train, args = (batch_num,))
        proc.start()
        proc.join()
        if 0 == proc.exitcode:
            batch_cnt += batch_num
            print "Finished %d random batches." %batch_cnt
    
