#! /usr/bin/env python
#-*- coding:utf-8 -*-

import os

if __name__ == '__main__':
    batch_num = 512
    for batch_no in range(0, 2496, batch_num):
        os.system('python train.py %d %d' %(batch_no, batch_num))

