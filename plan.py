#! /usr/bin/env python
#-*- coding:utf-8 -*-

from data_util import *
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


rnnlm_dir = os.path.join(save_dir, 'rnnlm')

if not os.path.exists(rnnlm_dir):
    os.mkdir(rnnlm_dir)

rnnlm_path = os.path.join(rnnlm_dir, 'model')


_int2word, _word2int = get_keywords()
_ranks = get_word_ranks()

_VOCAB_SIZE = len(_int2word)

_NUM_UNITS = 512
_NUM_LAYERS = 4
_BATCH_SIZE = 128
_NUM_EPOCHS = 1
_LEARN_RATE = 0.002
_DECAY_RATE = 0.97


def _encode_words(words):
    return [_word2int[word] for word in words]

def _cut(sentence):
    words = []
    toks = list(jieba.cut(sentence))
    for tok in toks:
        if len(tok) < 4:
            words.append(tok)
        else:
            words.extend(segment(tok))
    return words

def _extract(sentence):
    return filter(lambda x: x in _word2int, _cut(sentence))

def _batch_train_data(_train_data, epoch):
    i = 0
    while i < len(_train_data):
        j = min(len(_train_data), i+_BATCH_SIZE)
        vects = [_encode_words(words) for words in _train_data[i:j]]
        lengths = [len(vect)-1 for vect in vects]
        max_len = max(lengths)
        for k, vect in enumerate(vects):
            if len(vect) < max_len:
                vects[k].extend([_VOCAB_SIZE-1]*(max_len-len(vect)))
        print "[Training RNNML] epoch = %d/%d, processing %d/%d ..." \
                %(epoch, _NUM_EPOCHS, j, len(_train_data))
        yield np.matrix(vects), np.array(lengths)
        i = j


class Planner:

    def __init__(self):
        with tf.variable_scope('planner'):
            softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, _VOCAB_SIZE])
            softmax_b = tf.get_variable('softmax_b', [_VOCAB_SIZE])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding",
                        initializer = tf.random_uniform([_VOCAB_SIZE, _NUM_UNITS], -1.0, 1.0))
         
        cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
        self.init_state = cell.zero_state(_BATCH_SIZE, tf.float32)
        self.inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.seq_length = tf.placeholder(tf.int32, [_BATCH_SIZE])

        outputs, self.final_state = tf.nn.dynamic_rnn(cell,
                initial_state = self.init_state,
                inputs = tf.nn.embedding_lookup(embedding, self.inputs),
                sequence_length = self.seq_length,
                scope = 'planner')

        output = tf.reshape(outputs, [-1, _NUM_UNITS])
        self.logits = tf.nn.bias_add(tf.matmul(output, softmax_w), bias = softmax_b)

        self.targets = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,
                labels = tf.one_hot(tf.reshape(self.targets, [-1]), depth = _VOCAB_SIZE))
        total_loss = tf.reduce_mean(loss)
        self.opt_op = tf.train.AdadeltaOptimizer(learning_rate = _LEARN_RATE,
                rho = _DECAY_RATE).minimize(total_loss)

    def train(self, sess):
        print "Start training RNNML ..."
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
        sess.run(init_op)
        _train_data = get_lm_train_data()
        for epoch in range(_NUM_EPOCHS):
            for _batch, _lengths in _batch_train_data(_train_data, epoch):
                inputs = _batch[:,:-1]
                targets = _batch[:,1:]
                sess.run([self.opt_op], feed_dict = {
                    self.inputs: _batch[:,:-1],
                    self.targets: _batch[:,1:],
                    self.seq_length: _lengths})
            saver.save(sess, rnnlm_path)
        print "RNNML has been generated."

    def _expand(self, words, num):
        def _get_probable_idx(vals, vect):
            idx = 0
            for i in range(1, _VOCAB_SIZE-1):   # exclude '^' and '$'
                if i not in vect and (idx == 0 or vals[i] > vals[idx]):
                    idx = i
            return idx
        words.insert(0, u'^')
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(rnnlm_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                self.train(sess)
                ckpt = tf.train.get_checkpoint_state(rnnlm_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            vect = _encode_words(words)
            inputs = np.full((_BATCH_SIZE, num+1), _VOCAB_SIZE-1, np.int32)
            inputs[0,:len(vect)] = vect
            seq_length = np.zeros(_BATCH_SIZE, dtype=np.int32)
            seq_length[0] = len(vect)
            logits, state = sess.run([self.logits, self.final_state], feed_dict = {
                self.inputs: inputs, self.seq_length: seq_length})
            vect.append(_get_probable_idx(logits[-1], vect))
            words.append(_int2word[vect[-1]])
            inputs[0,:].fill(_VOCAB_SIZE-1)
            seq_length[0] = 1
            while len(words) <= num:
                inputs[0,0] = vect[-1]
                logits, state = sess.run([self.logits, self.final_state], feed_dict = {
                    self.inputs: inputs[:,:1],
                    self.seq_length: seq_length,
                    self.init_state: state})
                vect.append(_get_probable_idx(logits[-1], vect))
                words.append(_int2word[vect[-1]])

    def plan(self, text):
        keywords = sorted(reduce(lambda x,y:x+y, map(_extract, split_sentences(text))),
            cmp = lambda x,y: cmp(_ranks[x], _ranks[y]))
        words = [keywords[idx] for idx in \
                filter(lambda i: 0==i or keywords[i]!=keywords[i-1], range(len(keywords)))]
        if len(words) < 4:
            self._expand(words, 4)
        elif len(words) > 4 and len(words) < 8:
            self._expand(words, 8)
        else:
            while len(words) > 8:
                words.pop()
        return words


if __name__ == '__main__':
    planner = Planner()
    with tf.Session() as sess:
        planner.train(sess)

