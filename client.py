#! /usr/bin/env python
#-*- coding:utf-8 -*-

import random

from data_util import *

n_words = 4

if __name__ == '__main__':
    corpus = Corpus()
    model = corpus.get_model()
    jieba.analyse.set_stop_words(stop_path)
    while True:
        line = raw_input('Test:\t').decode('utf-8').strip()
        if line.lower() == 'quit' or line.lower() == 'exit':
            break
        elif len(line) > 0:
            sentences = split_sentences(line)
            keywords = []
            for sentence in sentences:
                keywords.extend(corpus.extract_keywords(sentence))
            if len(keywords) < n_words:
                queues = [[] for sentence in sentences]
                for i, sentence in enumerate(sentences):
                    queues[i].extend(jieba.analyse.extract_tags(sentence,
                            allowPOS=('ns', 'n', 'nr', 't')))
                    queues[i].extend(list(jieba.cut(sentence)))
                flag = True
                while flag:
                    flag = False
                    for i, q in enumerate(queues):
                        if len(q) > 0:
                            flag = True
                            word = q[0]
                            queues[i] = q[1:]
                            if word not in keywords:
                                keywords.append(word)
            '''
            if len(keywords) < n_words and len(keywords) > 0:
                uprint(keywords)
                keywords.extend([result[0] for result in \
                        model.most_similar(positive=keywords, topn=n_words-len(keywords))])
            '''
            if len(keywords) < n_words:
                corpus_kw = corpus.get_keywords()
                while len(keywords) < n_words:
                    word = corpus_kw[random.randint(0, len(corpus_kw)-1)]
                    if word not in keywords:
                        keywords.append(word)
            keywords = keywords[:n_words]
            print "Keywords:\t",
            for word in keywords:
                print repr(word).decode('unicode-escape'),
            print 
