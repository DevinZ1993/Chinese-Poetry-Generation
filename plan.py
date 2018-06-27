#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from data_utils import gen_train_data
from gensim import models
from paths import save_dir, plan_data_path, check_uptodate
from random import random, shuffle
from rank_words import RankedWords
from singleton import Singleton
from utils import split_sentences, NUM_OF_SENTENCES
import jieba
import os


_plan_model_path = os.path.join(save_dir, 'plan_model.bin')


def train_planner():
    # TODO: try other keyword-expansion models.
    print("Training Word2Vec-based planner ...")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not check_uptodate(plan_data_path):
        gen_train_data()
    word_lists = []
    with open(plan_data_path, 'r') as fin:
        for line in fin.readlines():
            word_lists.append(line.strip().split('\t'))
    model = models.Word2Vec(word_lists, size = 512, min_count = 5)
    model.save(_plan_model_path)


class Planner(Singleton):

    def __init__(self):
        self.ranked_words = RankedWords()
        if not os.path.exists(_plan_model_path):
            train_planner()
        self.model = models.Word2Vec.load(_plan_model_path)

    def plan(self, text):
        return self._expand(self._extract(text))

    def _expand(self, keywords):
        if len(keywords) < NUM_OF_SENTENCES:
            filtered_keywords = list(filter(lambda w : w in \
                    self.model.wv,  keywords))
            if len(filtered_keywords) > 0:
                similars = self.model.wv.most_similar(
                        positive = filtered_keywords)
                # Sort similar words in decreasing similarity with randomness.
                similars = sorted(similars, key = lambda x: x[1] * random())
                for similar in similars:
                    keywords.add(similar[0])
                    if len(keywords) == NUM_OF_SENTENCES:
                        break
            prob_sum = sum(1. / (i + 1) \
                    for i, word in enumerate(self.ranked_words) \
                    if word not in keywords)
            rand_val = prob_sum * random()
            word_idx = 0
            s = 0
            while len(keywords) < NUM_OF_SENTENCES \
                    and word_idx < len(self.ranked_words):
                word = self.ranked_words[word_idx]
                s += 1.0 / (word_idx + 1)
                if word not in keywords and rand_val < s:
                    keywords.add(word)
                word_idx += 1
        results = list(keywords)
        shuffle(results)
        return results

    def _extract(self, text):
        def extract_from_sentence(sentence):
            return filter(lambda w : w in self.ranked_words,
                jieba.lcut(sentence))
        keywords = set()
        for sentence in split_sentences(text):
            keywords.update(extract_from_sentence(sentence))
        return keywords


# For testing purpose.
if __name__ == '__main__':
    planner = Planner()
    keywords = planner.plan("春天到了，桃花开了。")
    print(keywords)

