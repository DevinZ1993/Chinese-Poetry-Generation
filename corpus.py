#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from rhyme import RhymeDict


_corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
        'yuan.all', 'ming.all', 'qing.all']


def _parse_corpus(raw_file, json_file):
    print "Parsing %s ..." %raw_file ,
    sys.stdout.flush()
    rdict = RhymeDict()
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
                for sentence in poem['sentences']:
                    if not reduce(lambda x,ch: x and rdict.has_char(ch), sentence, True):
                        flag = False
                        break
                if flag:
                    data.append(poem)
            line = fin.readline().strip()
    with codecs.open(json_file, 'w', 'utf-8') as fout:
        json.dump(data, fout)
    print "Done (%d poems)" %len(data)
    return data


def get_all_corpus():
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


if __name__ == '__main__':
    corpus = get_all_corpus()
    print "Size of the entire corpus: %d" % len(corpus)

