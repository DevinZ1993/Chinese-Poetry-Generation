#! /usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import sys
import os
import json
import jieba

raw_dir = 'raw'
data_dir = 'data'
save_dir = 'save'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# ========== ShiXueHanYing ==========

sxhy_raw = os.path.join(raw_dir, 'shixuehanying.txt')
sxhy_path = os.path.join(data_dir, 'sxhy_dict.txt')

def _gen_sxhy_dict():
    sxhy_dict = dict()
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
                        sxhy_dict[tok] = tag
            line = fin.readline().strip()
    with codecs.open(sxhy_path, 'w', 'utf-8') as fout:
        for word in sxhy_dict:
            fout.write(word+'\n')

if not os.path.exists(sxhy_path):
    _gen_sxhy_dict()
jieba.load_userdict(sxhy_path)

def _get_sxhy_dict():
    _sxhy_dict = set()
    with codecs.open(sxhy_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            _sxhy_dict.add(line.strip())
            line = fin.readline()
    return _sxhy_dict

_sxhy_dict = _get_sxhy_dict()


# ========== utility functions ===========

def uprint(x):
    print repr(x).decode('unicode-escape'),

def uprintln(x):
    print repr(x).decode('unicode-escape')

def is_CN_char(ch):
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def split_sentences(line):
    sentences = []
    i = 0
    for j in range(len(line)+1):
        if j == len(line) or line[j] in [u'，', u'。', u'！', u'。', u'、']:
            if i < j:
                sentence = u''.join(filter(is_CN_char, line[i:j]))
                sentences.append(sentence)
            i = j+1
    return sentences

def segment(sentence):
    if len(sentence) < 3:
        return [sentence]
    else:
        segs = []
        for i in range(0, len(sentence), 2):
            if i+3 == len(sentence):
                if sentence[i:i+2] in _sxhy_dict and sentence[i+1:] not in _sxhy_dict:
                    segs.extend([sentence[i:i+2], sentence[i+2:]])
                elif sentence[i:i+2] not in _sxhy_dict and sentence[i+1:] in _sxhy_dict:
                    segs.extend([sentence[i:i+1], sentence[i+1:]])
                else:
                    segs.extend(list(jieba.cut(sentence[i:])))
                break
            else:
                segs.append(sentence[i:i+2])
        return segs


# ========== raw stopwords ==========

stopwords_raw = os.path.join(raw_dir, 'stopwords.txt')

def get_stopwords():
    stopwords = set()
    with codecs.open(stopwords_raw, 'r', 'utf-8') as fin:
        line = fin.readline().strip()
        while line:
            stopwords.add(line)
            line = fin.readline().strip()
    return stopwords


# ========== poem corpus ==========

_corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
        'yuan.all', 'ming.all', 'qing.all']

def _is_poem(poem):
    if poem['source'] == 'qsc_tab.txt':
        return False
    else:
        sentences = split_sentences(poem['body'])
        return len(sentences) >= 4 and \
                (len(sentences[0]) == 5 or len(sentences[0]) == 7) and \
                reduce(lambda x, sentence: x and len(sentence) == len(sentences[0]) and \
                reduce(lambda y, ch: y and is_CN_char(ch), sentence, True),
                        sentences[1:], True)

def _parse_corpus(raw_file, json_file):
    print "Parsing %s ..." %raw_file ,
    sys.stdout.flush()
    data = []
    with codecs.open(raw_file, 'r', 'utf-8') as fin:
        tags = fin.readline().strip().split(u'\t')
        line = fin.readline().strip()
        while line:
            toks = line.split(u'\t')
            poem = {'source':os.path.basename(raw_file)}
            for idx, tok in enumerate(toks):
                poem[tags[idx]] = tok
            flag = True
            left = poem['body'].find(u'（')
            while left >= 0:
                right = poem['body'].find(u'）')
                if right < left:
                    flag = False
                    uprintln(poem['body'])
                    break
                else:
                    poem['body'] = poem['body'][:left]+poem['body'][right+1:]
                    left = poem['body'].find(u'（')
            if flag and poem['body'].find(u'）') < 0:
                data.append(poem)
            line = fin.readline().strip()
    with codecs.open(json_file, 'w', 'utf-8') as fout:
        json.dump(data, fout)
    print "Done (%d poems)" %len(data)
    return data

def _get_all_corpus():
    corpus = []
    for raw in _corpus_list:
        json_file = os.path.join(data_dir, raw.replace('all', 'json').replace('txt', 'json'))
        try:
            with codecs.open(json_file, 'r', 'utf-8') as fin:
                data = json.load(fin)
        except IOError:
            data = _parse_corpus(os.path.join(raw_dir, raw), json_file)
        finally:
            corpus.extend(data)
    return corpus

def get_poems():
    return [poem['body'] for poem in filter(_is_poem, _get_all_corpus())]


# =========== words over all poems ==========

words_path = os.path.join(data_dir, 'word_ranks.json')

def _rank_all_words():
    corpus = _get_all_corpus()
    stopwords=  get_stopwords()
    print "Start TextRank over the entire corpus ..."
    adjlist = dict()
    for idx, poem in enumerate(corpus):
        if 0 == (idx+1)%10000:
            print "[TextRank] Scanning %d/%d poems ..." %(idx+1, len(corpus))
        sentences = split_sentences(poem['body'])
        for sentence in sentences:
            segs = filter(lambda seg: seg not in stopwords, segment(sentence))
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
    damp = 0.85
    scores = dict((word,1.0) for word in adjlist)
    for i in xrange(1200):
        print "[TextRank] Start iteration %d ..." %i,
        sys.stdout.flush()
        cnt = 0
        new_scores = dict()
        for word in adjlist:
            new_scores[word] = (1-damp)+damp*sum(adjlist[other][word]*scores[other] for other in adjlist[word])
            if scores[word] != new_scores[word]:
                cnt += 1
        print "Done (%d/%d)" %(cnt, len(scores))
        if 0 == cnt:
            break
        else:
            scores = new_scores
    words = sorted([(word,score) for word,score in scores.items()],
            cmp = lambda x,y: cmp(y[1], x[1]))
    with codecs.open(words_path, 'w', 'utf-8') as fout:
        json.dump(words, fout)
    print "TextRank is done."

def get_word_ranks():
    if not os.path.exists(words_path):
        _rank_all_words()
    with codecs.open(words_path, 'r', 'utf-8') as fin:
        items = json.load(fin)
    return dict((item[0], idx) for idx, item in enumerate(items))


# ========== vocabulary over selected poems ==========

vocab_path = os.path.join(data_dir, 'vocab.json')

def _gen_vocab(poem_list):
    vocab = set()
    for poem in poem_list:
        sentences = split_sentences(poem)
        for sentence in sentences:
            for ch in sentence:
                vocab.add(ch)
    int2ch = list(vocab)
    with codecs.open(vocab_path, 'w', 'utf-8') as fout:
        json.dump(int2ch, fout)

def get_vocab():
    if not os.path.exists(vocab_path):
        _gen_vocab(get_poems())
    with codecs.open(vocab_path, 'r', 'utf-8') as fin:
        int2ch = json.load(fin)   
    int2ch.insert(0, u'^')
    int2ch.append(u' ')
    ch2int = dict((ch, i) for i, ch in enumerate(int2ch))
    return int2ch, ch2int


# ========== training data for seq2seq ==========

train_path = os.path.join(data_dir, 'train.txt')

def _gen_train_data():
    poems = get_poems()
    ranks = get_word_ranks()
    stopwords = get_stopwords()
    print "Generating training data ..."
    data = []
    for idx, poem in enumerate(poems):
        sentences = split_sentences(poem)
        if len(sentences) == 4 or len(sentences) == 8:
            flag = True
            lines = u''
            context = u''
            for sentence in sentences:
                row = [sentence]
                segs = filter(lambda seg: seg not in stopwords, segment(sentence))
                if 0 == len(segs):
                    flag = False
                    break
                keyword = reduce(lambda x,y: x if ranks[x]<=ranks[y] else y, segs)
                row.append(keyword)
                row.append(context)
                context += sentence+' '
            if flag:
                data.append(row)
        if 0 == (idx+1)%10000:
            print "[Training Data] %d/%d poems are processed." %((idx+1), len(poems))
    data = sorted(data, cmp = lambda x,y: cmp(len(x[0]), len(y[0])) if len(x[0]) != len(y[0])\
            else cmp(len(x[-1]), len(y[-1])))
    with codecs.open(train_path, 'w', 'utf-8') as fout:
        for row in data:
            fout.write(row[0]+'\t'+row[1]+'\t'+row[2]+'\n')
    print "Training data is generated."

def get_train_data():
    if not os.path.exists(train_path):
        _gen_train_data()
    data = []
    with codecs.open(train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            toks = line.strip().split('\t')
            data.append({'sentence':toks[0],
                'keyword':toks[1],
                'context':toks[2] if len(toks) == 3 else u''})
            line = fin.readline()
    return data


# =========== training data for RNNLM ==========

lm_train_path = os.path.join(data_dir, 'lm_train.txt')
keywords_path = os.path.join(data_dir, 'keywords.json')

def _gen_lm_train_data():
    poems = get_poems()
    ranks = get_word_ranks()
    stopwords = get_stopwords()
    print "Generating training data for RNNLM ..."
    data = []
    all_words = set()
    for idx, poem in enumerate(poems):
        keywords = []
        flag = True
        sentences = split_sentences(poem)
        if len(sentences) <= 8:
            for sentence in sentences:
                segs = filter(lambda seg: seg not in stopwords, segment(sentence))
                if 0 == len(segs):
                    flag = False
                    break
                keyword = reduce(lambda x,y: x if ranks[x]<=ranks[y] else y, segs)
                keywords.append(keyword)
                all_words.add(keyword)
            if flag:
                data.append(keywords)
        if 0 == (idx+1)%10000:
            print "[RNNLM Training Data] %d/%d poems are processed." %((idx+1), len(poems))
    data = sorted(data, cmp = lambda x,y: cmp(len(x), len(y)))
    with codecs.open(lm_train_path, 'w', 'utf-8') as fout:
        for keywords in data:
            for keyword in keywords:
                fout.write(keyword+'\t')
            fout.write('\n')
    with codecs.open(keywords_path, 'w', 'utf-8') as fout:
        json.dump(list(all_words), fout)
    print "Training data for RNNLM is generated."

def get_lm_train_data():
    if not os.path.exists(lm_train_path):
        _gen_lm_train_data()
    data = []
    with codecs.open(lm_train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            data.append([u'^'])
            data[-1].extend(line.strip().split('\t'))
            line = fin.readline()
    return data

def get_keywords():
    if not os.path.exists(keywords_path):
        _gen_lm_train_data()
    with codecs.open(keywords_path, 'r', 'utf-8') as fin:
        int2word = json.load(fin)
    int2word.insert(0, u'^')
    int2word.append(u' ')
    word2int = dict((word, idx) for idx, word in enumerate(int2word))
    return int2word, word2int


if __name__ == '__main__':
    if not os.path.exists(words_path):
        _rank_all_words()
    if not os.path.exists(train_path):
        _gen_train_data()
    if not os.path.exists(lm_train_path) or not os.path.exists(keywords_path):
        _gen_lm_train_data()

