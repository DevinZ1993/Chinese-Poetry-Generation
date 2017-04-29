#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *


py_raw = os.path.join(raw_dir, 'pinyin.txt')

_rhy_path = os.path.join(data_dir, 'rhy_dict.json')


_VOWELS = ['A', 'O', 'E', 'I', 'U', 'V']


def _get_vowels(pinyin):
    i = len(pinyin)-1
    while i >= 0 and pinyin[i] in _VOWELS:
        i -= 1
    return pinyin[i+1:]

def _get_rhyme(pinyin):
    vowels = _get_vowels(pinyin)
    if vowels in ['A', 'IA', 'UA']:
        return 1
    elif vowels in ['O', 'E', 'UO']:
        return 2
    elif vowels in ['IE', 'VE']:
        return 3
    elif vowels in ['AI', 'UAI']:
        return 4
    elif vowels in ['EI', 'UI']:
        return 5
    elif vowels in ['AO', 'IAO']:
        return 6
    elif vowels in ['OU', 'IU']:
        return 7
    elif vowels in ['AN', 'IAN', 'UAN', 'VAN']:
        return 8
    elif vowels in ['EN', 'IN', 'UN', 'VN']:
        return 9
    elif vowels in ['ANG', 'IANG', 'UANG']:
        return 10
    elif vowels in ['ENG', 'ING']:
        return 11
    elif vowels in ['ONG', 'IONG']:
        return 12
    elif (vowels == 'I' and not pinyin[0] in ['Z', 'C', 'S', 'R']) \
            or vowels == 'V':
        return 13
    elif vowels == 'I':
        return 14
    elif vowels == 'U':
        return 15
    else:
        return 0

def _gen_rhy_dict():
    ch2rhy = dict()
    with codecs.open(py_raw, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            toks = filter(lambda x: len(x) > 0, line.strip().split(' '))
            ch = unichr(int(toks[0], 16))
            if is_CN_char(ch):
                ch2rhy[ch] = (toks[1][:-1], int(toks[1][-1]))
            line = fin.readline()
    with codecs.open(_rhy_path, 'w', 'utf-8') as fout:
        json.dump(ch2rhy, fout)


class RhymeDict:

    def __init__(self):
        if not os.path.exists(_rhy_path):
            _gen_rhy_dict()
        with codecs.open(_rhy_path, 'r', 'utf-8') as fin:
            self.ch2rhy = json.load(fin)

    def has_char(self, ch):
        return ch in self.ch2rhy

    def get_rhyme(self, ch):
        return self.ch2rhy[ch][0]

    def get_tone(self, ch):
        if 1 <= self.ch2rhy[ch][1] <= 2:
            return 'p'
        elif 3 <= self.ch2rhy[ch][1] <= 4:
            return 'z'
        else:
            return None


class RhymeEvaluator:

    def __init__(self):
        self.rdict = RhymeDict()

    def eval(self, sentences):
        if len(sentences) != 4 or (len(sentences[0]) != 5 \
                and len(sentences[0]) != 7):
            return 0.
        else:
            for sentence in sentences[1:]:
                if len(sentence) != len(sentences[0]):
                    return 0.
            def _diff_tone(a, b):
                t1 = self.rdict.get_tone(a)
                t2 = self.rdict.get_tone(b)
                return ('p' == t1 and 'z' == t2) or ('z' == t1 and 'p' == t2)
            def _same_tone(a, b):
                t1 = self.rdict.get_tone(a)
                t2 = self.rdict.get_tone(b)
                return t1 and t1 == t2
            score = .1 if _diff_tone(sentences[0][1], sentences[0][3]) else .0
            for i, sentence in enumerate(sentences[1:]):
                if 0 == i%2:
                    if _diff_tone(sentence[3], sentences[i][3]):
                        score += .1
                else:
                    if _same_tone(sentence[3], sentences[i][3]):
                        score += .1
                if _diff_tone(sentence[1], sentence[3]):
                    score += .1
            rhyme = self.rdict.get_rhyme(sentences[1][-1])
            if rhyme > 0:
                if 'p' == self.rdict.get_tone(sentences[1][-1]):
                    score += .1
                if 'z' == self.rdict.get_tone(sentences[2][-1]) and \
                        rhyme != self.rdict.get_rhyme(sentences[2][-1]):
                    score += .1
                if 'p' == self.rdict.get_tone(sentences[3][-1]) and \
                        rhyme == self.rdict.get_rhyme(sentences[3][-1]):
                    score += .1
            return score

