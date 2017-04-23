#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import *
from rank_words import get_word_ranks
from data_utils import *
import tensorflow as tf
from tensorflow.contrib import rnn
from random import randint


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

rnnlm_dir = os.path.join(save_dir, 'rnnlm')

if not os.path.exists(rnnlm_dir):
    os.mkdir(rnnlm_dir)

_model_path = os.path.join(rnnlm_dir, 'model')


_NUM_UNITS = 512
_NUM_LAYERS = 4
_BATCH_SIZE = 128
_NUM_EPOCHS = 50
_LEARN_RATE = 0.002
_DECAY_RATE = 0.97


class Planner:

    def __init__(self):
        self.int2word, self.word2int = get_vocab()
        self.ranks =  get_word_ranks()

        with tf.variable_scope('planner'):
            softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, VOCAB_SIZE])
            softmax_b = tf.get_variable('softmax_b', [VOCAB_SIZE])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding",
                        initializer = tf.random_uniform([VOCAB_SIZE, _NUM_UNITS], -1.0, 1.0))
         
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
                labels = tf.one_hot(tf.reshape(self.targets, [-1]), depth = VOCAB_SIZE))
        total_loss = tf.reduce_mean(loss)
        self.opt_op = tf.train.AdadeltaOptimizer(learning_rate = _LEARN_RATE,
                rho = _DECAY_RATE).minimize(total_loss)

    @staticmethod
    def _get_saver(sess):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(rnnlm_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return saver
        
    def train(self, sess):
        print "Start training RNNML ..."
        saver = Planner._get_saver(sess)
        try:
            for epoch in range(_NUM_EPOCHS):
                batch_no = 0
                for w_mat, w_len in get_batch_kw_train_data(_BATCH_SIZE):
                    print "[Training RNNLM] epoch = %d/%d, line %d to %d ..." \
                            %(epoch, _NUM_EPOCHS, batch_no, batch_no+_BATCH_SIZE-1)
                    sess.run([self.opt_op], feed_dict = {
                        self.inputs: w_mat[:,:-1],
                        self.targets: w_mat[:,1:],
                        self.seq_length: w_len})
                    batch_no += _BATCH_SIZE
                    if 0 == batch_no/_BATCH_SIZE%32:
                        saver.save(sess, _model_path)
                        print "[Training RNNLM] Temporary model is saved."
                saver.save(sess, _model_path)
            print "RNNML has been generated."
        except KeyboardInterrupt:
            print "\nRNNML training is interrupted."

    def expand(self, words, num):
        def _get_probable_idx(vals, vect):
            idx = 0
            for i in range(1, VOCAB_SIZE-1):   # exclude '^' and '$'
                if i not in vect and (idx == 0 or vals[i] > vals[idx]):
                    idx = i
            return idx
        words.insert(0, u'^')
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(rnnlm_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                self.train(sess)
            _ = Planner._get_saver(sess)
            vect = encode_words(self.word2int, words)
            w_mat = fill_np_matrix([vect], _BATCH_SIZE, VOCAB_SIZE-1)
            w_len = fill_np_array([len(vect)], _BATCH_SIZE, 0)
            logits, state = sess.run([self.logits, self.final_state], feed_dict = {
                self.inputs: w_mat, self.seq_length: w_len})
            vect.append(_get_probable_idx(logits[-1], vect))
            words.append(self.int2word[vect[-1]])
            w_len[0] = 1
            while len(words) <= num:
                w_mat[0,:].fill(VOCAB_SIZE-1)
                w_mat[0,0] = vect[-1]
                logits, state = sess.run([self.logits, self.final_state], feed_dict = {
                    self.inputs: w_mat[:,:1],
                    self.seq_length: w_len,
                    self.init_state: state})
                vect.append(_get_probable_idx(logits[-1], vect))
                words.append(self.int2word[vect[-1]])
        del words[0]

    def plan(self, text):
        def extract(sentence):
            return filter(lambda x: x in self.ranks, list(jieba.cut(sentence)))
        keywords = sorted(reduce(lambda x,y:x+y, map(extract, split_sentences(text))),
            cmp = lambda x,y: cmp(self.ranks[x], self.ranks[y]))
        words = [keywords[idx] for idx in \
                filter(lambda i: 0==i or keywords[i]!=keywords[i-1], range(len(keywords)))]
        if len(words) < 4:
            self.expand(words, 4)
        else:
            while len(words) > 4:
                words.pop()
        return words


if __name__ == '__main__':
    planner = Planner()
    kw_train_data = get_kw_train_data()
    for row in kw_train_data:
        num = randint(1,3)
        uprint(row[1:])
        print "num = %d" %num
        guess = row[1:num+1]
        planner.expand(guess, 4)
        uprintln(guess)
        print

