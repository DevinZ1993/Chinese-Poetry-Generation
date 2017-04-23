#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter
from vocab import *
from quatrains import get_quatrains
from rank_words import get_word_ranks
import numpy as np


train_path = os.path.join(data_dir, 'train.txt')
kw_train_path = os.path.join(data_dir, 'kw_train.txt')


def _gen_train_data():
    segmenter = Segmenter()
    poems = get_quatrains()
    ranks = get_word_ranks()
    print "Generating training data ..."
    data = []
    kw_data = []
    for idx, poem in enumerate(poems):
        sentences = poem['sentences']
        if len(sentences) == 4:
            flag = True
            lines = u''
            context = u''
            rows = []
            kw_row = []
            for sentence in sentences:
                rows.append([sentence])
                segs = filter(lambda seg: seg in ranks, segmenter.segment(sentence))
                if 0 == len(segs):
                    flag = False
                    break
                keyword = reduce(lambda x,y: x if ranks[x] < ranks[y] else y, segs)
                kw_row.append(keyword)
                rows[-1].append(keyword)
                rows[-1].append(context)
                context += sentence+' '
            if flag:
                data.extend(rows)
                kw_data.append(kw_row)
        if 0 == (idx+1)%10000:
            print "[Training Data] %d/%d poems are processed." %(idx+1, len(poems))
    data = sorted(data, cmp = lambda x,y: cmp(len(x[0]), len(y[0])) if len(x[0]) != len(y[0])\
            else cmp(len(x[-1]), len(y[-1])))
    with codecs.open(train_path, 'w', 'utf-8') as fout:
        for row in data:
            fout.write(row[0]+'\t'+row[1]+'\t'+row[2]+'\n')
    with codecs.open(kw_train_path, 'w', 'utf-8') as fout:
        for row in kw_data:
            fout.write('\t'.join(row)+'\n')
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
                'context':toks[2] if len(toks) == 3 else ''})
            line = fin.readline()
    return data


def get_kw_train_data():
    if not os.path.exists(kw_train_path):
        _gen_train_data()
    data = []
    with codecs.open(kw_train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            data.append([u'^'])
            data[-1].extend(line.strip().split('\t'))
            line = fin.readline()
    return data


def fill_np_matrix(vects, batch_size, dummy):
    max_len = max(len(vect) for vect in vects)
    res = np.full([batch_size, max_len], dummy, dtype = np.int32)
    for row, vect in enumerate(vects):
        res[row, :len(vect)] = vect
    return res


def fill_np_array(vect, batch_size, dummy):
    res = np.full([batch_size], dummy, dtype = np.int32)
    res[:len(vect)] = vect
    return res

def get_train_dir_path(batch_size):
    return os.path.join(data_dir, 'train_%d'%batch_size)

def get_train_file_path(batch_size, idx):
    return os.path.join(get_train_dir_path(batch_size), '%d.json'%idx)

def _gen_batch_train_data(batch_size):
    print "Start to generate batch_train_data with batch_size = %d" %batch_size
    segmenter = Segmenter()
    _, word2int = get_vocab()
    data = get_train_data()
    os.mkdir(get_train_dir_path(batch_size))
    for i in range(0, len(data), batch_size):
        print "Generating batch_train_data starting from line %d  ..." %i
        text_data = data[i : min(len(data), i+batch_size)]
        json_path = get_train_file_path(batch_size, i)
        json_data = {'length': len(text_data)}
        try:
            json_data['s_lst'] = map(lambda x: encode_sentence(word2int, segmenter, x['sentence']),
                    text_data)
            json_data['k_lst'] = map(lambda x: encode_keyword(word2int, x['keyword']), text_data)
            json_data['c_lst'] = map(lambda x: encode_context(word2int, segmenter, x['context']),
                    text_data)
        except KeyError as e:
            uprintln(e)
        with codecs.open(json_path, 'w', 'utf-8') as fout:
            json.dump(json_data, fout)
    print "Finished generating batch_train_data."


def get_batch_train_data(batch_size):
    json_dir = get_train_dir_path(batch_size)
    if not os.path.exists(json_dir):
        _gen_batch_train_data(batch_size)
    idx = 0
    while True:
        json_path = get_train_file_path(batch_size, idx)
        if not os.path.exists(json_path):
            break
        else:
            with codecs.open(json_path, 'r', 'utf-8') as fin:
                json_data = json.load(fin)
            s_mat = fill_np_matrix(json_data['s_lst'], batch_size, VOCAB_SIZE-1)
            k_mat = fill_np_matrix(json_data['k_lst'], batch_size, VOCAB_SIZE-1)
            c_mat = fill_np_matrix(json_data['c_lst'], batch_size, VOCAB_SIZE-1)
            k_len = fill_np_array(map(len, json_data['k_lst']), batch_size, 0)
            c_len = fill_np_array(map(len, json_data['c_lst']), batch_size, 0)
            yield s_mat, k_mat, c_mat, k_len, c_len
            idx += json_data['length']


def get_batch_kw_train_data(batch_size):
    _, word2int = get_vocab()
    with codecs.open(kw_train_path, 'r', 'utf-8') as fin:
        i = 0
        while True:
            data = []
            j = i
            while j < i+batch_size:
                line = fin.readline().strip()
                if not line:
                    break
                else:
                    data.append(line.split('\t'))
                    j += 1
            if len(data) == 0:
                break
            else:
                w_mat = fill_np_matrix(map(lambda words: [0]+encode_words(word2int, words), data), 
                        batch_size, VOCAB_SIZE-1)
                w_len = fill_np_array(map(len, data), batch_size, 0)
                yield w_mat, w_len


if __name__ == '__main__':
    train_data = get_train_data()
    print "Size of the training data: %d" %len(train_data)
    kw_train_data = get_kw_train_data()
    print "Size of the keyword training data: %d" %len(kw_train_data)
    assert len(train_data) == 4*len(kw_train_data)


