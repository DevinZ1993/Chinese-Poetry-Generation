#! /usr/bin/env python
#-*- coding:utf-8 -*-

from data_util import *
from collections import deque
import math
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model_dir = os.path.join(save_dir,'model')

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = os.path.join(model_dir, 'model')


_int2ch, _ch2int = get_vocab()

_VOCAB_SIZE = len(_int2ch)

_NUM_UNITS = 512
_BATCH_SIZE = 128
_NUM_EPOCHS = 1
_LEARN_RATE = 0.002
_DECAY_RATE = 0.97


def encode_sentence(sentence):
    return [_ch2int[ch] for ch in sentence]

def _batch_train_data(start_batch_no = 0, batch_num = (1<<31)-1):
    batch_no = 0
    for sentences, keywords, contexts in batch_train_data(_BATCH_SIZE):
        if batch_no >= start_batch_no:
            s_matrix = fill_np_matrix(map(encode_sentence, sentences), _BATCH_SIZE, _VOCAB_SIZE-1)
            k_matrix = fill_np_matrix(map(encode_sentence, keywords), _BATCH_SIZE, _VOCAB_SIZE-1)
            c_matrix = fill_np_matrix(map(encode_sentence, contexts), _BATCH_SIZE, _VOCAB_SIZE-1)
            k_length = fill_np_array(map(len, keywords), _BATCH_SIZE, 0)
            c_length = fill_np_array(map(len, contexts), _BATCH_SIZE, 0)
            yield s_matrix, k_matrix, c_matrix, k_length, c_length
        batch_no += 1
        if batch_no >= start_batch_no + batch_num:
            break


class TrieNode:

    def __init__(self):
        self.freq = 0
        self.nexts = dict()

    def _add(self, word, freq, i):
        if i == len(word):
            self.freq += freq
        else:
            if word[i] not in self.nexts:
                self.nexts[word[i]] = TrieNode()
            self.nexts[word[i]]._add(word, freq, i+1)

    def add(self, word, freq):
        self._add(word, freq, 0)

    def get_freq(self, word):
        node = self
        for ch in word:
            if ch not in node.nexts:
                return 0
            else:
                node = node.nexts[ch]
        return node.freq

    def dfs(self, max_depth):
        results = []
        if self.freq > 0:
            results.append('')
        if max_depth > 0:
            for ch, node in self.nexts.items():
                results.extend(map(lambda x:ch+x, node.dfs(max_depth-1)))
        return results
        

def _build_trie():
    root = TrieNode()
    freqs = get_word_freqs()
    for word, freq in freqs.items():
        root.add(word, freq)
    return root

_root = _build_trie()

_BEAM_SIZE = 10

def _decode(rule, probs, length, words, p_words):
    heads = map(lambda x:_int2ch[x], sorted(range(1,_VOCAB_SIZE-1),
            cmp = lambda x,y: cmp(probs[y], probs[x])))
    def _word_filter(word):
        return len(word) <= length and (length == 3 or len(word) <= 2) \
                and rule.check(word) and word not in p_words \
                and reduce(lambda x,ch: x and \
                        reduce(lambda y,w: y and ch not in w, words, True),
                        word, True)
    for i in range(0, len(heads), _BEAM_SIZE):
        results = []
        for j in range(i, min(len(heads), i+_BEAM_SIZE)):
            if heads[j] in _root.nexts:
                results.extend(filter(_word_filter, 
                    [heads[j]+word for word in _root.nexts[heads[j]].dfs(min(3, length))]))
        freqs = dict((word, _root.get_freq(word)) for word in results)
        if len(results) > 0:
            result = results[0]
            for x in results[1:]:
                if freqs[x] > freqs[result]:
                    result = x
            rule.accept(result)
            words.append(result)
            return result


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
        self.r_final_state_fw, self.r_final_state_bw = r_final_state

        self.h_cell_fw = rnn.GRUCell(_NUM_UNITS)
        self.h_init_state_fw = tf.placeholder(tf.float32, [_BATCH_SIZE, _NUM_UNITS])
        self.h_inputs_fw = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.h_seq_length_fw = tf.placeholder(tf.int32, [_BATCH_SIZE])
        _, self.h_final_state_fw = tf.nn.dynamic_rnn(self.h_cell_fw,
                inputs = tf.nn.embedding_lookup(embedding, self.h_inputs_fw),
                sequence_length = self.h_seq_length_fw,
                initial_state = self.h_init_state_fw,
                scope = 'h_encoder_fw')

        self.h_cell_bw = rnn.GRUCell(_NUM_UNITS)
        self.h_init_state_bw = tf.placeholder(tf.float32, [_BATCH_SIZE, _NUM_UNITS])
        self.h_inputs_bw = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.h_seq_length_bw = tf.placeholder(tf.int32, [_BATCH_SIZE])
        _, self.h_final_state_bw = tf.nn.dynamic_rnn(self.h_cell_bw,
                inputs = tf.nn.embedding_lookup(embedding, self.h_inputs_bw),
                sequence_length = self.h_seq_length_bw,
                initial_state = self.h_init_state_bw,
                scope = 'h_encoder_bw')

        self.s_cell = rnn.GRUCell(_NUM_UNITS)
        self.s_init_state = tf.placeholder(tf.float32, [_BATCH_SIZE, _NUM_UNITS])
        self.s_inputs = tf.placeholder(tf.int32, [_BATCH_SIZE])
        self.attention_states = tf.placeholder(tf.float32, [_BATCH_SIZE, None, 2*_NUM_UNITS])
        s_outputs, self.s_final_state = legacy_seq2seq.embedding_attention_decoder(\
                decoder_inputs = [self.s_inputs],
                initial_state = self.s_init_state,
                attention_states = self.attention_states,
                cell = self.s_cell,
                num_symbols = _VOCAB_SIZE,
                embedding_size = _NUM_UNITS,
                num_heads = 1,
                output_size = None,
                feed_previous = False,
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

        h_states_fw = deque([sess.run(self.h_cell_fw.zero_state(_BATCH_SIZE, tf.float32))])
        for i in range(c_mat.shape[1]):
            h_states_fw.append(sess.run(self.h_final_state_fw, feed_dict = {
                self.h_init_state_fw: h_states_fw[-1], 
                self.h_inputs_fw: c_mat[:,i:i+1],
                self.h_seq_length_fw: _get_h_length(i)}))

        h_states_bw = [sess.run(self.h_cell_bw.zero_state(_BATCH_SIZE, tf.float32))]
        for i in range(c_mat.shape[1]-1, -1, -1):
            h_states_bw.append(sess.run(self.h_final_state_bw, feed_dict = {
                self.h_init_state_bw: h_states_bw[-1],
                self.h_inputs_bw: c_mat[:,i:i+1],
                self.h_seq_length_bw: _get_h_length(i)}))

        h_states = [np.concatenate([r_state_fw, r_state_bw], axis=1)]
        for _ in range(c_mat.shape[1]):
            h_states.append(np.concatenate([h_states_fw.popleft(), h_states_bw.pop()], axis=1))
        return np.stack(h_states, axis=1)

    def train(self, sess, batch_no = 0, batch_num = (1<<31)-1):
        print "Start training attention-based RNN enc-dec ..."
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)
        try:
            max_batch_no = batch_no + batch_num
            for s_mat, k_mat, c_mat, k_len, c_len in _batch_train_data(batch_no, batch_num):
                print "[Training Seq2Seq] Processing line %d to %d ..." %(batch_no*_BATCH_SIZE, (batch_no+1)*_BATCH_SIZE)
                attention_states = self._get_attention_states(sess, k_mat, c_mat, k_len, c_len)
                s_inputs = sess.run(tf.zeros([_BATCH_SIZE], tf.int32))
                s_state = sess.run(self.s_cell.zero_state(_BATCH_SIZE, tf.float32))
                for i in range(s_mat.shape[1]):
                    s_state, _ = sess.run([self.s_final_state, self.opt_op], feed_dict = {
                        self.s_inputs : s_inputs,
                        self.s_init_state: s_state,
                        self.attention_states: attention_states,
                        self.s_targets: s_mat[:,i:i+1]})
                    s_inputs = s_mat[:,i:i+1].reshape([-1])
                batch_no += 1
                if 0 == batch_no%128:
                    saver.save(sess, model_path)
                    print "[Training Seq2Seq] Temporary model is saved."
                if batch_no >= max_batch_no:
                    break
            print "Training has finished."
        except KeyboardInterrupt:
            print "\nTraining is interrupted."

    def _gen_sentence(self, sess, keyword, context, rule, p_words):
        k_mat = fill_np_matrix([encode_sentence(keyword)], _BATCH_SIZE, _VOCAB_SIZE-1)
        c_mat = fill_np_matrix([encode_sentence(context)], _BATCH_SIZE, _VOCAB_SIZE-1)
        k_len = fill_np_array([len(keyword)], _BATCH_SIZE, 0)
        c_len = fill_np_array([len(context)], _BATCH_SIZE, 0)
        attention_states = self._get_attention_states(sess, k_mat, c_mat, k_len, c_len)
        s_inputs = sess.run(tf.zeros([_BATCH_SIZE], tf.int32))
        s_state = sess.run(self.s_cell.zero_state(_BATCH_SIZE, tf.float32))
        words = []
        idx = 0
        while idx < rule.n_chars:
            s_state, _logits = sess.run([self.s_final_state, self.logits], feed_dict = {
                self.s_inputs: s_inputs,
                self.s_init_state: s_state,
                self.attention_states: attention_states})
            word = _decode(rule, _logits.tolist()[0], rule.n_chars-idx, words, p_words)
            _mat = fill_np_matrix([encode_sentence(word)], _BATCH_SIZE, _VOCAB_SIZE-1)
            for i in range(len(word)-1):
                s_state, _ = sess.run([self.s_final_state, self.logits], feed_dict = {
                    self.s_inputs: _mat[:,i].reshape([-1]),
                    self.s_init_state: s_state,
                    self.attention_states: attention_states})
            s_inputs = _mat[:,-1].reshape([-1])
            idx += len(word)
        p_words.extend(words)
        return u''.join(words)

    def generate(self, keywords):
        rule = RhymeRule()
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                self.train(sess)
                ckpt = tf.train.get_checkpoint_state(model_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            context = u''
            p_words = []
            for keyword in keywords:
                sentence = self._gen_sentence(sess, keyword, context, rule, p_words)
                uprintln(sentence)
                context += sentence+' '
            
if __name__ == '__main__':
    generator = Generator()
    for rows in batch_lm_train_data(1):
        uprintln(rows[0][1:])
        generator.generate(rows[0][1:])

