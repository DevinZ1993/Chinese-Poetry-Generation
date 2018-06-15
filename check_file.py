#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from utils import data_dir, save_dir
import os


sxhy_path = os.path.join(data_dir, 'sxhy_dict.txt')
char_dict_path = os.path.join(data_dir, 'char_dict.txt')
poems_path = os.path.join(data_dir, 'poem.txt')
char2vec_path = os.path.join(data_dir, 'char2vec.npy')
wordrank_path = os.path.join(data_dir, 'wordrank.txt')
plan_data_path = os.path.join(data_dir, 'plan_data.txt')
gen_data_path = os.path.join(data_dir, 'gen_data.txt')


# TODO: configure dependencies in another file.
_dependency_dict = {
        poems_path : [char_dict_path],
        char2vec_path : [char_dict_path, poems_path],
        wordrank_path : [sxhy_path, poems_path],
        gen_data_path : [char_dict_path, poems_path, sxhy_path, char2vec_path],
        plan_data_path : [char_dict_path, poems_path, sxhy_path, char2vec_path],
        }

def file_uptodate(path):
    if not os.path.exists(path):
        # File not found.
        return False
    timestamp = os.path.getmtime(path)
    if path in _dependency_dict:
        for dependency in _dependency_dict[path]:
            if not os.path.exists(dependency) or \
                    os.path.getmtime(dependency) > timestamp:
                # File stale.
                return False
    return True
