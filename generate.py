#! /usr/bin/env python
#-*- coding:utf-8 -*-

from data_util import *
from collections import deque
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq


model_dir = os.path.join(save_dir,'model')

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = os.path.join(model_dir, 'model')


_int2ch, _ch2int = get_vocab()

_VOCAB_SIZE = len(_int2ch)

_NUM_UNITS = 512
_BATCH_SIZE = 128
_NUM_EPOCHS = 100
_LEARN_RATE = 0.002
_DECAY_RATE = 0.97


def encode_sentence(sentence):
    return [_ch2int[ch] for ch in sentence]

def _to_np_matrix(vects):
    max_len = max(len(vect) for vect in vects)
    mat = np.full([_BATCH_SIZE, max_len], _VOCAB_SIZE-1, dtype = np.int32)
    for row, vect in enumerate(vects):
        mat[row, :len(vect)] = vect
    return mat

def _to_np_length(lengths):
    vect = np.full([_BATCH_SIZE], 0, dtype = np.int32)
    vect[:len(lengths)] = lengths
    return vect

def _batch_train_data(_train_data, epoch):
    i = 0
    while i < len(_train_data):
        j = i
        while j < len(_train_data) and j < i+_BATCH_SIZE and \
                len(_train_data[j]['sentence']) == len(_train_data[i]['sentence']):
            j += 1
        sentences = [row['sentence'] for row in _train_data[i:j]]
        keywords = [row['keyword'] for row in _train_data[i:j]]
        contexts = [row['context'] for row in _train_data[i:j]]
        k_length = _to_np_length(map(len, keywords))
        c_length = _to_np_length(map(len, contexts))
        s_matrix = _to_np_matrix(map(encode_sentence, sentences))
        k_matrix = _to_np_matrix(map(encode_sentence, keywords))
        c_matrix = _to_np_matrix(map(encode_sentence, contexts))
        print "[Training] epoch = %d/%d, processing %d/%d ..." \
                %(epoch, _NUM_EPOCHS, j, len(_train_data))
        yield s_matrix, k_matrix, c_matrix, k_length, c_length
        i = j

class TrieNode:
    
    def __init__(self):
        self.nexts = [None]*_VOCAB_SIZE
        self.end = False

    def add(self, word, i):
        if i == len(word):
            self.end = True
        else:
            j = _ch2int[word[i]]
            if not self.nexts[j]:
                self.nexts[j] = TrieNode()
            self.nexts[j].add(word, i+1)

_ranks = get_word_ranks()


def _beam_search(logits):
    def _most_prob_idxs(probs, size):
        return sorted(range(1,_VOCAB_SIZE-1),
                cmp = lambda x,y: cmp(probs[y], probs[x]))[:min(_VOCAB_SIZE-2, size)]
    sentence = u''
    i = 0
    '''
    while i+3 < len(logits):
        probs = dict()
        size = 8
        while 0 == len(probs):
            first_idxs = _most_prob_idxs(logits[i], size)
            second_idxs = _most_prob_idxs(logits[i+1], size)
            for first_idx in first_idxs:
                for second_idx in second_idxs:
                    word = _int2ch[first_idx]+_int2ch[second_idx]
                    #uprintln(word)
                    if word in _ranks:
                        probs[word] = math.log(logits[i][first_idx])+math.log(logits[i+1][second_idx])
            size <<= 1
        sentence += reduce(lambda x,y: x if x[1]>=y[1] else y,
                [(word, prob) for word, prob in probs.items()])[0]
        i += 2
    '''
    while i < len(logits):
        sentence += _int2ch[np.argmax(logits[i][1:-1])+1]
        i += 1
    return sentence


class Generator:

    def __init__(self):
        with tf.variable_scope('generator'):
            softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, _VOCAB_SIZE])
            softmax_b = tf.get_variable('softmax_b', [_VOCAB_SIZE])
            with tf.device("/cpu:0"):
                 embedding = tf.get_variable('embedding',
                         initializer = tf.random_uniform([_VOCAB_SIZE, _NUM_UNITS], -1.0, 1.0))

        r_cell_fw = rnn.GRUCell(_NUM_UNITS)
        r_cell_bw = rnn.GRUCell(_NUM_UNITS)
        r_init_state_fw = r_cell_fw.zero_state(_BATCH_SIZE, tf.float32)
        r_init_state_bw = r_cell_bw.zero_state(_BATCH_SIZE, tf.float32)
        self.r_inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.r_seq_length = tf.placeholder(tf.int32, [_BATCH_SIZE])
        _, r_final_state = tf.nn.bidirectional_dynamic_rnn(r_cell_fw, r_cell_bw,
                inputs = tf.nn.embedding_lookup(embedding, self.r_inputs),
                sequence_length = self.r_seq_length,
                initial_state_fw = r_init_state_fw,
                initial_state_bw = r_init_state_bw,
                scope = 'r_encoder')
        self.r_final_state_fw = r_final_state[0]
        self.r_final_state_bw = r_final_state[1]

        h_cell_fw = rnn.GRUCell(_NUM_UNITS)
        self.h_init_state_fw = h_cell_fw.zero_state(_BATCH_SIZE, tf.float32)
        self.h_inputs_fw = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.h_seq_length_fw = tf.placeholder(tf.int32, [_BATCH_SIZE])
        _, self.h_final_state_fw = tf.nn.dynamic_rnn(h_cell_fw,
                inputs = tf.nn.embedding_lookup(embedding, self.h_inputs_fw),
                sequence_length = self.h_seq_length_fw,
                initial_state = self.h_init_state_fw,
                scope = 'h_encoder_fw')

        h_cell_bw = rnn.GRUCell(_NUM_UNITS)
        self.h_init_state_bw = h_cell_bw.zero_state(_BATCH_SIZE, tf.float32)
        self.h_inputs_bw = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.h_seq_length_bw = tf.placeholder(tf.int32, [_BATCH_SIZE])
        _, self.h_final_state_bw = tf.nn.dynamic_rnn(h_cell_bw,
                inputs = tf.nn.embedding_lookup(embedding, self.h_inputs_bw),
                sequence_length = self.h_seq_length_bw,
                initial_state = self.h_init_state_bw,
                scope = 'h_encoder_bw')

        s_cell = rnn.GRUCell(_NUM_UNITS)
        self.s_init_state = s_cell.zero_state(_BATCH_SIZE, tf.float32)
        s_inputs = tf.zeros([_BATCH_SIZE], dtype = tf.int32)
        self.attention_states = tf.placeholder(tf.float32, [_BATCH_SIZE, None, 2*_NUM_UNITS])
        s_outputs, self.s_final_state = legacy_seq2seq.embedding_attention_decoder(\
                decoder_inputs = [s_inputs],
                initial_state = self.s_init_state,
                attention_states = self.attention_states,
                cell = s_cell,
                num_symbols = _VOCAB_SIZE,
                embedding_size = _NUM_UNITS,
                num_heads = 1,
                output_size = None,
                feed_previous = True,
                update_embedding_for_previous = False,
                scope = 's_decoder')
        self.logits = tf.nn.bias_add(tf.matmul(s_outputs[0], softmax_w), bias = softmax_b)

        self.s_targets = tf.placeholder(tf.int32, [_BATCH_SIZE,1])
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.s_targets, [-1])],
                [tf.ones([_BATCH_SIZE], dtype = tf.float32)])
        total_loss = tf.reduce_mean(loss)
        self.opt_op = tf.train.AdadeltaOptimizer(learning_rate = _LEARN_RATE,
                rho = _DECAY_RATE).minimize(total_loss)

    def _get_attention_states(self, sess, k_mat, c_mat, k_len, c_len):
        r_state_fw, r_state_bw = sess.run([self.r_final_state_fw, self.r_final_state_bw], feed_dict = {
            self.r_inputs: k_mat, self.r_seq_length: k_len})

        def _get_h_length(i):
            return np.array(map(lambda j: 1 if i<=c_len[j] else 0, range(c_len.shape[0])),
                    dtype = np.int32)

        h_states_fw = deque([sess.run(self.h_init_state_fw)])
        for i in range(c_mat.shape[1]):
            h_states_fw.append(sess.run(self.h_final_state_fw, feed_dict = {
                self.h_init_state_fw: h_states_fw[-1], 
                self.h_inputs_fw: c_mat[:,i:i+1],
                self.h_seq_length_fw: _get_h_length(i)}))

        h_states_bw = [sess.run(self.h_init_state_bw)]
        for i in range(c_mat.shape[1]-1, -1, -1):
            h_states_bw.append(sess.run(self.h_final_state_bw, feed_dict = {
                self.h_init_state_bw: h_states_bw[-1],
                self.h_inputs_bw: c_mat[:,i:i+1],
                self.h_seq_length_bw: _get_h_length(i)}))

        h_states = [np.concatenate([r_state_fw, r_state_bw], axis=1)]
        for _ in range(c_mat.shape[1]):
            h_states.append(np.concatenate([h_states_fw.popleft(), h_states_bw.pop()], axis=1))
        return np.stack(h_states, axis=1)

    def train(self, sess):
        print "Start training attention-based RNN enc-dec ..."
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
        sess.run(init_op)
        _train_data = get_train_data()[:100]
        _train_data = filter(lambda x:len(x['sentence'])==5, _train_data)
        for epoch in range(_NUM_EPOCHS):
            for s_mat, k_mat, c_mat, k_len, c_len in _batch_train_data(_train_data, epoch):
                attention_states = self._get_attention_states(sess, k_mat, c_mat, k_len, c_len)
                s_state = sess.run(self.s_init_state)
                for i in range(s_mat.shape[1]):
                    s_state, _ = sess.run([self.s_final_state, self.opt_op], feed_dict = {
                        self.s_init_state: s_state,
                        self.attention_states: attention_states,
                        self.s_targets: s_mat[:,i:i+1]})
            saver.save(sess, model_path)
        print "Training has finished."

    def _gen_sentence(self, sess, keyword, context, n_chars):
        k_mat = np.full([_BATCH_SIZE, len(keyword)], _VOCAB_SIZE-1, dtype=np.int32) 
        k_mat[0] = encode_sentence(keyword)
        k_len = np.zeros(_BATCH_SIZE, dtype=np.int32)
        k_len[0] = len(keyword)
        c_mat = np.full([_BATCH_SIZE, len(context)], _VOCAB_SIZE-1, dtype=np.int32)
        c_mat[0] = encode_sentence(context)
        c_len = np.zeros(_BATCH_SIZE, dtype=np.int32)
        c_len[0] = len(context)
        attention_states = self._get_attention_states(sess, k_mat, c_mat, k_len, c_len)
        s_state = sess.run(self.s_init_state)
        logits = []
        for _ in range(n_chars):
            s_state, _logits = sess.run([self.s_final_state, self.logits], feed_dict = {
                self.s_init_state: s_state,
                self.attention_states: attention_states})
            logits.append(_logits.tolist()[0])
        return _beam_search(logits)


    def generate(self, keywords):
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                print "No available checkpoint ..."
                self.train(sess)
                ckpt = tf.train.get_checkpoint_state(model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            n_chars = 7
            context = u''
            for keyword in keywords:
                sentence = self._gen_sentence(sess, keyword, context, n_chars)
                uprintln(sentence)
                context += sentence+' '
            
if __name__ == '__main__':
    generator = Generator()
    with tf.Session() as sess:
        generator.train(sess)

