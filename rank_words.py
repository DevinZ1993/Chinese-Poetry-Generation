#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import raw_dir, wordrank_path, check_uptodate
from poems import Poems
from segment import Segmenter
from singleton import Singleton
import json
import os
import sys


_stopwords_path = os.path.join(raw_dir, 'stopwords.txt')


_damp = 0.85


def _get_stopwords():
    stopwords = set()
    with open(_stopwords_path, 'r') as fin:
        for line in fin.readlines():
            stopwords.add(line.strip())
    return stopwords


# TODO: try other keyword-extraction algorithms. This doesn't work well.

class RankedWords(Singleton):

    def __init__(self):
        self.stopwords = _get_stopwords()
        if not check_uptodate(wordrank_path):
            self._do_text_rank()
        with open(wordrank_path, 'r') as fin:
            self.word_scores = json.load(fin)
        self.word2rank = dict((word_score[0], rank) 
                for rank, word_score in enumerate(self.word_scores))

    def _do_text_rank(self):
        print("Do text ranking ...")
        adjlists = self._get_adjlists()
        print("[TextRank] Total words: %d" % len(adjlists))

        # Value initialization.
        scores = dict()
        for word in adjlists:
            scores[word] = [1.0, 1.0]

        # Synchronous value iterations.
        itr = 0
        while True:
            sys.stdout.write("[TextRank] Iteration %d ..." % itr)
            sys.stdout.flush()
            for word, adjlist in adjlists.items():
                scores[word][1] = (1.0 - _damp) + _damp * \
                        sum(adjlists[other][word] * scores[other][0] 
                                for other in adjlist)
            eps = 0
            for word in scores:
                eps = max(eps, abs(scores[word][0] - scores[word][1]))
                scores[word][0] = scores[word][1]
            print(" eps = %f" % eps)
            if eps <= 1e-6:
                break
            itr += 1

        # Dictionary-based comparison with TextRank score as a tie-breaker.
        segmenter = Segmenter()
        def cmp_key(x):
            word, score = x
            return (0 if word in segmenter.sxhy_dict else 1, -score)
        words = sorted([(word, score[0]) for word, score in scores.items()], 
                key = cmp_key)

        # Store ranked words and scores.
        with open(wordrank_path, 'w') as fout:
            json.dump(words, fout)

    def _get_adjlists(self):
        print("[TextRank] Generating word graph ...")
        segmenter = Segmenter()
        poems = Poems()
        adjlists = dict()
        # Count number of co-occurrence.
        for poem in poems:
            for sentence in poem:
                words = []
                for word in segmenter.segment(sentence):
                    if word not in self.stopwords:
                        words.append(word)
                for word in words:
                    if word not in adjlists:
                        adjlists[word] = dict()
                for i in range(len(words)):
                    for j in range(i + 1, len(words)):
                        if words[j] not in adjlists[words[i]]:
                            adjlists[words[i]][words[j]] = 1.0
                        else:
                            adjlists[words[i]][words[j]] += 1.0
                        if words[i] not in adjlists[words[j]]:
                            adjlists[words[j]][words[i]] = 1.0
                        else:
                            adjlists[words[j]][words[i]] += 1.0
        # Normalize weights.
        for a in adjlists:
            sum_w = sum(w for _, w in adjlists[a].items())
            for b in adjlists[a]:
                adjlists[a][b] /= sum_w
        return adjlists

    def __getitem__(self, index):
        if index < 0 or index >= len(self.word_scores):
            return None
        return self.word_scores[index][0]

    def __len__(self):
        return len(self.word_scores)

    def __iter__(self):
        return map(lambda x: x[0], self.word_scores)

    def __contains__(self, word):
        return word in self.word2rank

    def get_rank(self, word):
        if word not in self.word2rank:
            return len(self.word2rank)
        return self.word2rank[word]


# For testing purpose.
if __name__ == '__main__':
    ranked_words = RankedWords()
    for i in range(100):
        print(ranked_words[i])

