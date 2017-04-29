#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter, get_sxhy_dict
from quatrains import get_quatrains


stopwords_raw = os.path.join(raw_dir, 'stopwords.txt')

rank_path = os.path.join(data_dir, 'word_ranks.json')


def get_stopwords():
    stopwords = set()
    with codecs.open(stopwords_raw, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            stopwords.add(line.strip())
            line = fin.readline()
    return stopwords


def _text_rank(adjlist):
    damp = 0.85
    scores = dict((word,1.0) for word in adjlist)
    try:
        for i in xrange(100000):
            print "[TextRank] Start iteration %d ..." %i,
            sys.stdout.flush()
            cnt = 0
            new_scores = dict()
            for word in adjlist:
                new_scores[word] = (1-damp)+damp*sum(adjlist[other][word]*scores[other] \
                        for other in adjlist[word])
                if scores[word] != new_scores[word]:
                    cnt += 1
            print "Done (%d/%d)" %(cnt, len(scores))
            if 0 == cnt:
                break
            else:
                scores = new_scores
        print "TextRank is done."
    except KeyboardInterrupt:
        print "\nTextRank is interrupted."
    sxhy_dict = get_sxhy_dict()
    def _compare_words(a, b):
        if a[0] in sxhy_dict and b[0] not in sxhy_dict:
            return -1
        elif a[0] not in sxhy_dict and b[0] in sxhy_dict:
            return 1
        else:
            return cmp(b[1], a[1])
    words = sorted([(word,score) for word,score in scores.items()],
            cmp = _compare_words)
    with codecs.open(rank_path, 'w', 'utf-8') as fout:
        json.dump(words, fout)


def _rank_all_words():
    segmenter = Segmenter()
    stopwords = get_stopwords()
    print "Start TextRank over the selected quatrains ..."
    quatrains = get_quatrains()
    adjlist = dict()
    for idx, poem in enumerate(quatrains):
        if 0 == (idx+1)%10000:
            print "[TextRank] Scanning %d/%d poems ..." %(idx+1, len(quatrains))
        for sentence in poem['sentences']:
            segs = filter(lambda word: word not in stopwords,
                    segmenter.segment(sentence))
            for seg in segs:
                if seg not in adjlist:
                    adjlist[seg] = dict()
            for i, seg in enumerate(segs):
                for _, other in enumerate(segs[i+1:]):
                    if seg != other:
                        adjlist[seg][other] = adjlist[seg][other]+1 \
                                if other in adjlist[seg] else 1.0
                        adjlist[other][seg] = adjlist[other][seg]+1 \
                                if seg in adjlist[other] else 1.0
    for word in adjlist:
        w_sum = sum(weight for other, weight in adjlist[word].items())
        for other in adjlist[word]:
            adjlist[word][other] /= w_sum
    print "[TextRank] Weighted graph has been built."
    _text_rank(adjlist)


def get_word_ranks():
    if not os.path.exists(rank_path):
        _rank_all_words()
    with codecs.open(rank_path, 'r', 'utf-8') as fin:
        ranks = json.load(fin)
    return dict((pair[0], idx) for idx, pair in enumerate(ranks))


if __name__ == '__main__':
    ranks = get_word_ranks()
    print "Size of word_ranks: %d" % len(ranks)

