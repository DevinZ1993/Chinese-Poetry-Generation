#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from common import *
from segment import Segmenter
from poems import Poems
from rank_words import RankedWords
import subprocess
import argparse
import re


def gen_train_data():
    print("Generating training data ...")
    segmenter = Segmenter()
    poems = Poems()
    ranked_words = RankedWords()
    plan_data = []
    gen_data = []
    for poem in poems:
        if len(poem) != 4:
            continue # Only consider quatrains.
        valid = True
        context = ''
        gen_lines = []
        keywords = []
        for sentence in poem:
            words = list(filter(lambda seg: seg in ranked_words, 
                    segmenter.segment(sentence)))
            if len(words) == 0:
                valid = False
                break
            keyword = words[0]
            for word in words[1 : ]:
                if ranked_words.get_rank(word) < ranked_words.get_rank(keyword):
                    keyword = word
            gen_line = sentence + '$\t' + keyword + '\t' + context + '\n'
            gen_lines.append(gen_line)
            keywords.append(keyword)
            context += sentence + '$'
        if valid:
            plan_data.append('\t'.join(keywords) + '\n')
            gen_data.extend(gen_lines)
    with open(plan_data_path, 'w') as fout:
        for line in plan_data:
            fout.write(line)
    with open(gen_data_path, 'w') as fout:
        for line in gen_data:
            fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training data generation.')
    parser.add_argument('--clean', dest = 'clean', default = False, 
            action = 'store_true', help = 'clean all processed data')
    args = parser.parse_args()
    if args.clean:
        for f in os.listdir(data_dir):
            if not re.match('raw', f):
                print("Delete %s." % os.path.join(data_dir, f))
                os.remove(os.path.join(data_dir, f))
        subprocess.run(args=["./char2vec.py"], check = True, 
                stdout = sys.stdout)
        subprocess.run(args=["./rank_words.py"], check = True, 
                stdout = sys.stdout)
    gen_train_data()

