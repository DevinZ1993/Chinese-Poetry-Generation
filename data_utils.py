#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import gen_data_path, plan_data_path, check_uptodate
from poems import Poems
from rank_words import RankedWords
from segment import Segmenter
import re
import subprocess


def gen_train_data():
    print("Generating training data ...")
    segmenter = Segmenter()
    poems = Poems()
    poems.shuffle()
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


def batch_train_data(batch_size):
    """ Training data generator for the poem generator."""
    if not check_uptodate(gen_data_path):
        print("Warning: training data is not found!")
        gen_train_data()
    keywords = []
    contexts = []
    sentences = []
    with open(gen_data_path, 'r') as fin:
        for line in fin.readlines():
            toks = line.strip().split('\t')
            sentences.append(toks[0])
            keywords.append(toks[1])
            contexts.append(toks[2] if len(toks) == 3 else '')
            if len(keywords) == batch_size:
                yield keywords, contexts, sentences
                keywords.clear()
                contexts.clear()
                sentences.clear()
        if len(keywords) > 0:
            yield keywords, contexts, sentences


if __name__ == '__main__':
    if not check_uptodate(plan_data_path) or \
            not check_uptodate(gen_data_path):
        gen_train_data()

