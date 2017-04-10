#! /usr/bin/env python
#-*- coding:utf-8 -*-

import codecs
import json
import sys
import os
import re
import jieba
import jieba.analyse
from gensim import models

raw_dir = 'raw'
data_dir = 'data'
model_dir = 'save'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

stop_path = os.path.join(raw_dir, 'stopwords.txt')
sxhy_raw = os.path.join(raw_dir, 'shixuehanying.txt')

_raw_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
        'yuan.all', 'ming.all', 'qing.all']

dict_path = os.path.join(data_dir, 'dict.txt')
dict_tagged = os.path.join(data_dir, 'dict_tagged.json')

model_path = os.path.join(model_dir, 'gensim.model')

kw_json = os.path.join(data_dir, 'keywords.json')
train_path = os.path.join(data_dir, 'train.txt')


def uprint(x):
    print repr(x).decode('unicode-escape')

def is_CN_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def is_poem(sentences):
    return len(sentences) >= 4 and \
            (len(sentences[0]) == 5 or len(sentences[0]) == 7) and \
            reduce(lambda x, sentence: x and len(sentence) == len(sentences[0]) and \
            reduce(lambda y, ch: y and is_CN_char(ch), sentence, True),
                    sentences[1:], True)

def split_sentences(line):
    sentences = []
    i = 0
    for j in range(len(line)+1):
        if j == len(line) or line[j] in [u'，', u'。', u'！', u'。', u'、']:
            if i < j:
                sentences.append(line[i:j])
            i = j+1
    return sentences

def _select_poems(data):
    poems = []
    for poem in data:
        sentences = split_sentences(poem['body'])
        if 'music' not in poem and is_poem(sentences) and 4 == len(sentences):
            poems.append(poem)
    return poems

def _parse_corpus(raw_file, json_file):
    print "Parsing %s ..." %raw_file ,
    sys.stdout.flush()
    data = []
    with codecs.open(raw_file, 'r', 'utf-8') as fin:
        tags = fin.readline().strip().split(u'\t')
        line = fin.readline().strip()
        while line:
            toks = line.split(u'\t')
            entity = dict()
            for idx, tok in enumerate(toks):
                entity[tags[idx]] = tok
            data.append(entity)
            line = fin.readline().strip()
    with codecs.open(json_file, 'w', 'utf-8') as fout:
        json.dump(data, fout)
    print "Done (%d poems)" %len(data)
    return data

def _get_word_dict():
    try:
        with codecs.open(dict_tagged, 'r', 'utf-8') as fin:
            word_dict = json.load(fin)
    except Exception:
        print "Parsing ShiXueHanYing ..."
        word_dict = dict()
        with codecs.open(sxhy_raw, 'r', 'utf-8') as fin:
            line = fin.readline().strip()
            while line:
                if line.startswith('<begin>'):
                    tag = line.split('\t')[2]
                elif not line.startswith('<end>'):
                    toks = line.split('\t')
                    if len(toks) == 3:
                        toks = toks[2].split(' ')
                        tok_list = []
                        for tok in toks:
                            if len(tok) < 4:
                                tok_list.append(tok)
                            else:
                                tok_list.extend(list(jieba.cut(tok)))
                        for tok in tok_list:
                            word_dict[tok] = tag
                line = fin.readline().strip()
        with codecs.open(dict_path, 'w', 'utf-8') as fout:
            for word in word_dict:
                fout.write(word+'\n')
        with codecs.open(dict_tagged, 'w', 'utf-8') as fout:
            json.dump(word_dict, fout)
        print "User dictionary generated."
    finally:
        return word_dict


class Corpus:

    def __init__(self):
        self.word_dict = _get_word_dict()
        jieba.load_userdict(dict_path)

    def get_word_dict(self):
        return self.word_dict

    def get_data(self, truncate=True):
        try:
            return self.data
        except AttributeError:
            self.data = []
            for raw in _raw_list:
                json_file = os.path.join(data_dir, raw.replace('all', 'json').replace('txt', 'json'))
                try:
                    with codecs.open(json_file, 'r', 'utf-8') as fin:
                        data = json.load(fin)
                except Exception:
                    data = _parse_corpus(os.path.join(raw_dir, raw), json_file)
                finally:
                    self.data.extend(data)
            if truncate:
                self.data = _select_poems(self.data)[:10000]   # restricted to first 10000 poems
            return self.data

    def get_text(self, truncate=True):
        return '\n'.join(poem['body'] for poem in self.get_data(truncate))

    def get_model(self):
        try:
            return self.model
        except AttributeError:
            try:
                self.model = models.Word2Vec.load(model_path)
            except Exception:
                text = self.get_text(truncate=False)
                print "Generating gensim model ...",
                sys.stdout.flush()
                self.model = models.Word2Vec(text)
                self.model.save(model_path)
                print "Done"
            finally:
                return self.model

    def get_keywords(self):
        try:
            return self.keywords
        except AttributeError:
            try:
                with codecs.open(kw_json, 'r', 'utf-8') as fin:
                    self.keywords = json.load(fin)
            except Exception:
                text = self.get_text()
                print "Generating keywords ...",
                sys.stdout.flush()
                self.keywords = jieba.analyse.textrank(text, topK=None,
                        allowPOS=('ns', 'n', 'nr', 't'))
                with codecs.open(kw_json, 'w', 'utf-8') as fout:
                    json.dump(self.keywords, fout)
                print "Done"
            finally:
                return self.keywords

    def get_kdict(self):
        try:
            return self.kdict
        except AttributeError:
            keywords = self.get_keywords()
            self.kdict = dict((keyword, idx) for idx, keyword in enumerate(keywords))
            for word in self.word_dict:
                if len(word) > 1 and word not in self.kdict:
                    self.kdict[word] = len(self.kdict)
            return self.kdict

    def extract_keywords(self, sentence):
        kdict = self.get_kdict()
        keywords = set()
        for i in range(0, len(sentence)):
            for j in range(i+1, min(i+4, len(sentence))):
                if sentence[i:j] in kdict:
                    keywords.add(sentence[i:j])
        return sorted(list(keywords),
                cmp = lambda x, y: kdict[x]-kdict[y])

    def extract_keyword(self, sentence):
        keywords = self.extract_keywords(sentence)
        if len(keywords) > 0:
            return keywords[0]
        else:
            tags = jieba.analyse.extract_tags(sentence,
                    allowPOS=('ns', 'n', 'nr', 't'))
            return tags[0] if len(tags) > 0 else list(jieba.cut(sentence))[-1]


if __name__ == '__main__':
    corpus = Corpus()
    with codecs.open(train_path, 'w', 'utf-8') as fout:
        for poem in corpus.get_data():
            sentences = split_sentences(poem['body'])
            context = ''
            for sentence in sentences:
                fout.write(sentence+'\t')
                fout.write(corpus.extract_keyword(sentence)+'\t')
                fout.write(context+"\n")
                context += sentence+' '

