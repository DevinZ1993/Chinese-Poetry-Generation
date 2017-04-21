#! /usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import sys
import os
import json
import jieba
import random
import numpy as np

raw_dir = 'raw'
data_dir = 'data'
save_dir = 'save'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


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


# ========== ShiXueHanYing for segmentation ==========

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

def get_sxhy_dict():
    sxhy_dict = set()
    with codecs.open(sxhy_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            sxhy_dict.add(line.strip())
            line = fin.readline()
    return sxhy_dict

class Segmenter:

    def __init__(self):
        self.sxhy_dict = get_sxhy_dict()

    def segment(self, sentence):
        if len(sentence) < 3:
            return [sentence]
        else:
            segs = []
            for i in range(0, len(sentence), 2):
                if i+3 == len(sentence):
                    if sentence[i:i+2] in self.sxhy_dict and sentence[i+1:] not in self.sxhy_dict:
                        segs.extend([sentence[i:i+2], sentence[i+2:]])
                    elif sentence[i:i+2] not in self.sxhy_dict and sentence[i+1:] in self.sxhy_dict:
                        segs.extend([sentence[i:i+1], sentence[i+1:]])
                    else:
                        segs.extend(list(jieba.cut(sentence[i:])))
                    break
                elif sentence[i:i+2] in self.sxhy_dict:
                    segs.append(sentence[i:i+2])
                else:
                    segs.extend(jieba.cut(sentence[i:i+2]))
            #uprintln(segs)
            return segs


# ========== ShiSiYun for rhyme checking ==========

ssy_raw = os.path.join(raw_dir, 'shisiyun.txt')
py_raw = os.path.join(raw_dir, 'pinyin.txt')

rhy_path = os.path.join(data_dir, 'rhy_dict.txt')
    
def _get_ssy_ch2rhy():
    ch2rhy = dict()
    with codecs.open(ssy_raw, 'r', 'utf-8') as fin:
        yun = 0
        line = fin.readline()
        while line:
            if len(line.strip()) > 0:
                if is_CN_char(line[0]):
                    yun += 1
                    state = 0
                elif 0 == state:
                    line = line.strip()
                    if line.find(u'平') >= 0:
                        assert line.find(u'阴') >= 0 or line.find(u'阳') >= 0
                        tone = 0 if line.find(u'阴') >= 0 else 1
                    else:
                        assert line.find(u'上') >= 0 or line.find(u'去') >= 0
                        tone = 2 if line.find(u'上') >= 0 else 3
                    state = 1
                else:
                    flag = True
                    rhyme = (yun<<2)+tone
                    for ch in line:
                        if ch == u'(':
                            flag = False
                        elif ch == u')':
                            flag = True
                        elif flag and is_CN_char(ch):
                            if ch not in ch2rhy:
                                ch2rhy[ch] = []
                            if rhyme not in ch2rhy[ch]:
                                ch2rhy[ch].append(rhyme)
                    state = 0
            line = fin.readline()
    return ch2rhy
    
def _gen_rhy_dict():
    print "Generating rhyme dictionary ...",
    sys.stdout.flush()
    ch2rhy = _get_ssy_ch2rhy()
    p2ch = dict()
    with codecs.open(py_raw, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            toks = filter(lambda x: len(x) > 0, line.strip().split(' '))
            ch = unichr(int(toks[0], 16))
            if is_CN_char(ch):
                for key in toks[1:]:
                    if key not in p2ch:
                        p2ch[key] = []
                    p2ch[key].append(ch)
            line = fin.readline()
    p2rhy = dict()
    for key, chs in p2ch.items():
        if key not in p2rhy:
            for ch in chs:
                if ch in ch2rhy and len(ch2rhy[ch]) == 1:
                    pin = key[:-1]
                    rhyme = ch2rhy[ch][0]&(-1<<2)
                    for tone in range(4):
                        p2rhy[pin+str(tone+1)] = rhyme+tone
                    break
    for key, chs in p2ch.items():
        if key in p2rhy:
            for ch in chs:
                if ch not in ch2rhy:
                    ch2rhy[ch] = []
                if p2rhy[key] not in ch2rhy[ch]:
                    ch2rhy[ch].append(p2rhy[key])
    with codecs.open(rhy_path, 'w', 'utf-8') as fout:
        json.dump(ch2rhy, fout)
    print "Done."

rule_raw = os.path.join(raw_dir, 'tone_rules.json')

class RhymeDict:

    def __init__(self):
        if not os.path.exists(rhy_path):
            _gen_rhy_dict()
        with codecs.open(rhy_path, 'r', 'utf-8') as fin:
            self.ch2rhy = json.load(fin)

    def has_char(self, ch):
        return ch in self.ch2rhy

    def check_rhyme(self, ch, rhyme):
        for rhy in self.ch2rhy[ch]:
            if rhy&(-1<<2) == rhyme&(-1<<2):
                return True
        return False

    def check_tone(self, ch, pingze):
        if pingze.lower() == 'p':
            for rhy in self.ch2rhy[ch]:
                if (rhy & 3) < 2:
                    return True
            return False
        else:
            for rhy in self.ch2rhy[ch]:
                if (rhy & 3) >= 2:
                    return True
            return False

class RhymeRule:

    def __init__(self):
        self.rdict = RhymeDict()
        with codecs.open(rule_raw, 'r', 'utf-8') as fin:
            rules = json.load(fin)
        rule = rules[random.randint(0, len(rules)-1)]
        self.n_chars = rule['n_chars']
        self.tones = rule['tones']
        self.pos = 0
        self.lim = self.n_chars
        self.rhyme = None

    def check(self, word):
        if self.pos+len(word) > self.lim:
            return False
        else:
            idx = self.pos
            for ch in word:
                if not self.rdict.check_tone(ch, self.tones[idx]):
                    return False
                elif 'P' == self.tones[idx] and self.rhyme \
                        and not self.rdict.check_rhyme(ch, self.rhyme):
                    return False
                idx += 1
            return True

    def accept(self, word):
        self.pos += len(word)
        if self.pos == self.lim:
            self.lim += self.n_chars
            if self.tones[self.pos-1] == 'P' and not self.rhyme:
                self.rhyme = self.rdict.ch2rhy[word[-1]][0]


# ========== poem corpus ==========

_corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
        'yuan.all', 'ming.all', 'qing.all']

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
                if tags[idx] != 'body':
                    poem[tags[idx]] = tok
                else:
                    body = tok
            flag = True
            left = body.find(u'（')
            while left >= 0:
                right = body.find(u'）')
                if right < left:
                    flag = False
                    break
                else:
                    body = body[:left]+body[right+1:]
                    left = body.find(u'（')
            if flag and body.find(u'）') < 0:
                poem['sentences'] = split_sentences(body)
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

class PoemChecker:

    def __init__(self):
        self._rdict = RhymeDict()

    def is_poem(self, poem):
        if poem['source'] == 'qsc_tab.txt':
            return False
        else:
            sentences = poem['sentences']
            return len(sentences) >= 4 and \
                    (len(sentences[0]) == 5 or len(sentences[0]) == 7) and \
                    reduce(lambda x, sentence: x and len(sentence) == len(sentences[0]),
                            sentences[1:], True) and \
                    reduce(lambda x, sentence: x and reduce(lambda y, ch: y and self._rdict.has_char(ch),
                        sentence, True), sentences, True)

def _get_poems():
    corpus = _get_all_corpus()
    pc = PoemChecker()
    return filter(pc.is_poem, corpus)


# =========== stopwords for word ranking ==========

stopwords_raw = os.path.join(raw_dir, 'stopwords.txt')

def _get_stopwords():
    stopwords = set()
    with codecs.open(stopwords_raw, 'r', 'utf-8') as fin:
        line = fin.readline().strip()
        while line:
            stopwords.add(line)
            line = fin.readline().strip()
    return stopwords


# =========== word ranking over all corpus ==========

rank_path = os.path.join(data_dir, 'word_ranks.json')

def _text_rank(adjlist):
    damp = 0.85
    scores = dict((word,1.0) for word in adjlist)
    try:
        for i in xrange(5000):
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
    stopwords=  _get_stopwords()
    segmenter = Segmenter()
    rdict = RhymeDict()
    def _word_filter(word):
        return reduce(lambda x,ch: x and rdict.has_char(ch),
                word, word not in stopwords)
    print "Start TextRank over the entire corpus ..."
    corpus = _get_all_corpus()
    adjlist = dict()
    for idx, poem in enumerate(corpus):
        if 0 == (idx+1)%10000:
            print "[TextRank] Scanning %d/%d poems ..." %(idx+1, len(corpus))
        for sentence in poem['sentences']:
            segs = filter(_word_filter, segmenter.segment(sentence))
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
        items = json.load(fin)
    return dict((item[0], idx) for idx, item in enumerate(items))


# ========== training data for seq2seq ==========

train_path = os.path.join(data_dir, 'train.txt')

def _gen_train_data():
    poems = _get_poems()
    ranks = get_word_ranks()
    segmenter = Segmenter()
    print "Generating training data ..."
    data = []
    for idx, poem in enumerate(poems):
        sentences = poem['sentences']
        if len(sentences) == 4:
            flag = True
            lines = u''
            context = u''
            rows = []
            for sentence in sentences:
                rows.append([sentence])
                segs = filter(lambda seg: seg in ranks, segmenter.segment(sentence))
                if 0 == len(segs):
                    flag = False
                    break
                keyword = reduce(lambda x,y: x if ranks[x] < ranks[y] else y, segs)
                rows[-1].append(keyword)
                rows[-1].append(context)
                context += sentence+' '
            if flag:
                data.extend(rows)
        if 0 == (idx+1)%10000:
            print "[Training Data] %d/%d poems are processed." %((idx+1), len(poems))
    data = sorted(data, cmp = lambda x,y: cmp(len(x[0]), len(y[0])) if len(x[0]) != len(y[0])\
            else cmp(len(x[-1]), len(y[-1])))
    with codecs.open(train_path, 'w', 'utf-8') as fout:
        for row in data:
            fout.write(row[0]+'\t'+row[1]+'\t'+row[2]+'\n')
    print "Training data is generated."

def _get_train_data():
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


# ========== vocabulary over training data ==========

vocab_path = os.path.join(data_dir, 'vocab.json')

def _gen_vocab():
    print "Generating vocabulary from train.txt ...",
    train_data = _get_train_data()
    vocab = set()
    for row in train_data:
        for ch in row['sentence']:
            vocab.add(ch)
    int2ch = list(vocab)
    with codecs.open(vocab_path, 'w', 'utf-8') as fout:
        json.dump(int2ch, fout)
    print "Done."

def get_vocab():
    if not os.path.exists(vocab_path):
        _gen_vocab()
    with codecs.open(vocab_path, 'r', 'utf-8') as fin:
        int2ch = json.load(fin)   
    int2ch.insert(0, u'^')
    int2ch.append(u' ')
    ch2int = dict((ch, i) for i, ch in enumerate(int2ch))
    return int2ch, ch2int


# ========== word frequencies ===========

freq_path = os.path.join(data_dir, 'word_freqs.json')

def _gen_word_freqs():
    segmenter = Segmenter()
    _, ch2int = get_vocab()
    def _word_filter(word):
        return reduce(lambda x, ch: x and ch in ch2int,
                word, True)
    print "Start counting word frequencies ..."
    corpus = _get_all_corpus()
    freqs = dict()
    for idx, poem in enumerate(corpus):
        if 0 == (idx+1)%10000:
            print "[Word Freq] Scanning %d/%d poems ..." %(idx+1, len(corpus))
        for sentence in poem['sentences']:
            segs = filter(_word_filter, segmenter.segment(sentence))
            for seg in segs:
                freqs[seg] = 1 if seg not in freqs else freqs[seg]+1
    with codecs.open(freq_path, 'w', 'utf-8') as fout:
        json.dump(freqs, fout)
    print "Word freqs are obtained."

def get_word_freqs():
    if not os.path.exists(freq_path):
        _gen_word_freqs()
    with codecs.open(freq_path, 'r', 'utf-8') as fin:
        freqs = json.load(fin)
    return freqs


# =========== training data for RNNLM ==========

lm_train_path = os.path.join(data_dir, 'lm_train.txt')
keywords_path = os.path.join(data_dir, 'keywords.json')

def _gen_lm_train_data():
    poems = _get_poems()
    ranks = get_word_ranks()
    segmenter = Segmenter()
    print "Generating training data for RNNLM ..."
    data = []
    for idx, poem in enumerate(poems):
        if len(poem['sentences']) == 4:
            flag = True
            row = []
            for sentence in poem['sentences']:
                segs = filter(lambda seg: seg in ranks, segmenter.segment(sentence))
                if 0 == len(segs):
                    flag = False
                    break
                keyword = reduce(lambda x,y: x if ranks[x] < ranks[y] else y, segs)
                row.append(keyword)
            if flag:
                data.append(row)
        if 0 == (idx+1)%10000:
            print "[RNNLM Training Data] %d/%d poems are processed." %((idx+1), len(poems))
    data = sorted(data, cmp = lambda x,y: cmp(len(x), len(y)))
    keywords = set()
    with codecs.open(lm_train_path, 'w', 'utf-8') as fout:
        for row in data:
            for word in row:
                fout.write(word+'\t')
                keywords.add(word)
            fout.write('\n')
    with codecs.open(keywords_path, 'w', 'utf-8') as fout:
        json.dump(list(keywords), fout)
    print "Training data for RNNLM is generated."

def _get_lm_train_data():
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


# ========== data fetching in minibatch ==========

def fill_np_matrix(vects, batch_size, val):
    max_len = max(len(vect) for vect in vects)
    res = np.full([batch_size, max_len], val, dtype = np.int32)
    for row, vect in enumerate(vects):
        res[row, :len(vect)] = vect
    return res

def fill_np_array(vect, batch_size, val):
    res = np.full([batch_size], val, dtype = np.int32)
    res[:len(vect)] = vect
    return res

def _gen_batch_lines(path, batch_size):
    with codecs.open(path, 'r', 'utf-8') as fin:
        i = 0
        while True:
            j = i
            lines = []
            while j < i+batch_size:
                line = fin.readline().strip()
                if line:
                    lines.append(line)
                    j += 1
                else:
                    break
            if len(lines) > 0:
                yield lines
                i += len(lines)
            else:
                break

def batch_train_data(batch_size):
    if not os.path.exists(train_path):
        _gen_train_data()
    for lines in _gen_batch_lines(train_path, batch_size):
        toks = [line.split('\t') for line in lines]
        sentences = [line_toks[0] for line_toks in toks]
        keywords = [line_toks[1] for line_toks in toks]
        contexts = [line_toks[2] if len(line_toks) == 3 else u'' \
                for line_toks in toks]
        yield sentences, keywords, contexts

def batch_lm_train_data(batch_size):
    if not os.path.exists(lm_train_path):
        _gen_lm_train_data()
    for lines in _gen_batch_lines(lm_train_path, batch_size):
        yield [([u'^']+line.split('\t')) for line in lines]


if __name__ == '__main__':
    rdict = RhymeDict()
    print "Size of rhyme_dict: %d chars" %len(rdict.ch2rhy)
    ranks = get_word_ranks()
    print "Size of word_ranks: %d words" %len(ranks)
    corpus = _get_all_corpus()
    train_data = _get_train_data()
    lm_train_data = _get_lm_train_data()
    assert len(train_data) == 4*len(lm_train_data)
    print "Data set: %d quatrains from %d poems" \
            %(len(lm_train_data), len(corpus))
    int2ch, _ = get_vocab()
    print "Size of vocabulary: %d chars" %len(int2ch)
    word_freqs = get_word_freqs()
    print "Size of word_freqs: %d words" %len(word_freqs)
    word2int, _ = get_keywords()
    print "Size of keywords: %d words" %len(word2int)

