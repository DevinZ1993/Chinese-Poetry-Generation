#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *


ssy_raw = os.path.join(raw_dir, 'shisiyun.txt')
py_raw = os.path.join(raw_dir, 'pinyin.txt')
rules_raw = os.path.join(raw_dir, 'rhy_rules.txt')

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
    sys.stdout.flush()
    ch2rhy = _get_ssy_ch2rhy()
    p2ch = dict()
    ch2p = dict()
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
                ch2p[ch] = toks[1]
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
    for ch, p in ch2p.items():
        if p in p2rhy:
            if ch not in ch2rhy:
                ch2rhy[ch] = []
            if p2rhy[p] not in ch2rhy[ch]:
                ch2rhy[ch].append(p2rhy[p])
    with codecs.open(rhy_path, 'w', 'utf-8') as fout:
        json.dump(ch2rhy, fout)


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


def _read_rules():
    rules = []
    with open(rules_raw, 'r') as fin:
        line = fin.readline()
        while line:
            rule = dict()
            toks = line.strip().split('\t')
            rule['n_chars'] = int(toks[0])
            rule['tones'] = ''.join(toks[1:])
            rules.append(rule)
            line = fin.readline()
    return rules


class RhymeChecker:

    def __init__(self):
        self.rdict = RhymeDict()
        rules = _read_rules()
        rule = rules[random.randint(0, len(rules)-1)]
        self.n_chars = rule['n_chars']
        self.tones = rule['tones']
        self.pos = 0
        self.lim = 2
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
        if not self.check(word):
            return False
        else:
            self.pos += len(word)
            if self.pos == self.lim:
                if 0 == (self.lim+3)%self.n_chars:
                    self.lim += 3
                else:
                    self.lim += 2
                if self.tones[self.pos-1] == 'P' and not self.rhyme:
                    self.rhyme = self.rdict.ch2rhy[word[-1]][0]
            return True

