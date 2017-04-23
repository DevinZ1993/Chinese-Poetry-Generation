#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter
from corpus import get_all_corpus


vocab_path = os.path.join(data_dir, 'vocab.json')

VOCAB_SIZE = 50000


def _gen_vocab():
    print "Generating the vocabulary ..."
    segmenter = Segmenter()
    corpus = get_all_corpus()
    word_cnts = dict()
    for idx, poem in enumerate(corpus):
        for sentence in poem['sentences']:
            segs = segmenter.segment(sentence)
            for seg in segs:
                word_cnts[seg] = word_cnts[seg]+1 if seg in word_cnts else 1
        if 0 == (idx+1)%10000:
            print "[Gen Vocab] %d/%d poems have been processed." %(idx+1, len(corpus))
    vocab = sorted([word for word in word_cnts], key = lambda x: -word_cnts[x])[:VOCAB_SIZE-2]
    with codecs.open(vocab_path, 'w', 'utf-8') as fout:
        json.dump(vocab, fout)
    print "The global vocab is built."


def get_vocab():
    if not os.path.exists(vocab_path):
        _gen_vocab()
    int2word = [u'^']
    with codecs.open(vocab_path, 'r', 'utf-8') as fin:
        int2word.extend(json.load(fin))
    int2word.append(u' ')
    word2int = dict((word, idx) for idx, word in enumerate(int2word))
    return int2word, word2int


def encode_words(word2int, words):
    return map(lambda word: word2int[word], words)

def encode_sentence(word2int, segmenter, sentence):
    return encode_words(word2int, segmenter.segment(sentence))

def encode_keyword(word2int, keyword):
    return [word2int[keyword]]

def encode_context(word2int, segmenter, context):
    return reduce(lambda x,y:x+y, [encode_sentence(word2int, segmenter, sentence) \
            for sentence in context.split(' ')])

def decode_keyword(int2word, ranks, prob_list):
        return int2word(reduce(lambda x,y: x if prob_list[x] >= prob_list[y] else y,
            filter(lambda x: int2word[x] in ranks, range(1, VOCAB_SIZE-1))))

def decode_word(int2word, rhyme_checker, p_word, prob_list):
    words = map(lambda k: int2word[k],
            sorted(range(1, VOCAB_SIZE-1), key = lambda k:-prob_list[k]))
    for word in words:
        if word != p_word and rhyme_checker.accept(word):
            return word
    raise Exception('No proper words found!')

if __name__ == '__main__':
    int2word, _ = get_vocab()
    print "Size of the vocabulary: %d" % len(int2word)
    for word in int2word[:]:
        uprint(word)
    print

