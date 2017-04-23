#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import *
from rhyme import RhymeChecker
from data_utils import *
from collections import deque
from random import randint
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model_dir = os.path.join(save_dir,'seq2seq')

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

_model_path = os.path.join(model_dir, 'model')

_NUM_UNITS = 512
_BATCH_SIZE = 128
_NUM_EPOCHS = 50
_LEARN_RATE = 0.002
_DECAY_RATE = 0.97


class Generator:

    def __init__(self):
        with tf.variable_scope('generator'):
            softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, VOCAB_SIZE])
            softmax_b = tf.get_variable('softmax_b', [VOCAB_SIZE])
            with tf.device("/cpu:0"):
                 embedding = tf.get_variable('embedding',
                         initializer = tf.random_uniform([VOCAB_SIZE, _NUM_UNITS], -1.0, 1.0))

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
                num_symbols = VOCAB_SIZE,
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

    @staticmethod
    def _get_saver(sess):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
        else:
            saver.restore(sess, ckpt.model_checkpoint_path)
        return saver


    def _train_a_batch(self, sess, s_mat, k_mat, c_mat, k_len, c_len):
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

    def train(self, sess):
        print "Start training attention-based RNN enc-dec ..."
        saver = Generator._get_saver(sess)
        try:
            for epoch in range(_NUM_EPOCHS):
                batch_no = 0
                for s_mat, k_mat, c_mat, k_len, c_len in get_batch_train_data(_BATCH_SIZE):
                    print "[Training Seq2Seq] epoch = %d/%d, line %d to %d ..." \
                            %(epoch, _NUM_EPOCHS, batch_no*_BATCH_SIZE, (batch_no+1)*_BATCH_SIZE)
                    self._train_a_batch(sess, s_mat, k_mat, c_mat, k_len, c_len)
                    batch_no += 1
                    if 0 == batch_no%32:
                        saver.save(sess, _model_path)
                        print "[Training Seq2Seq] Temporary model is saved."
                saver.save(sess, _model_path)
            print "Training has finished."
        except KeyboardInterrupt:
            print "\nTraining is interrupted."

    def rand_train(self, sess, batch_num = 128):
        print "Start random training ..."
        saver = Generator._get_saver(sess)
        dir_path = get_train_dir_path(_BATCH_SIZE)
        files = os.listdir(dir_path)
        for batch_no in range(batch_num):
            file_path = os.path.join(dir_path, files[random.randint(0, len(files)-1)])
            print "[Training Seq2Seq] Processing file %s ..." %file_path
            with codecs.open(file_path, 'r', 'utf-8') as fin:
                json_data = json.load(fin)
            s_mat = fill_np_matrix(json_data['s_lst'], _BATCH_SIZE, VOCAB_SIZE-1)
            k_mat = fill_np_matrix(json_data['k_lst'], _BATCH_SIZE, VOCAB_SIZE-1)
            c_mat = fill_np_matrix(json_data['c_lst'], _BATCH_SIZE, VOCAB_SIZE-1)
            k_len = fill_np_array(map(len, json_data['k_lst']), _BATCH_SIZE, 0)
            c_len = fill_np_array(map(len, json_data['c_lst']), _BATCH_SIZE, 0)
            self._train_a_batch(sess, s_mat, k_mat, c_mat, k_len, c_len)
            if 0 == (batch_no+1)%32:
                saver.save(sess, _model_path)
                print "[Training Seq2Seq] Temporary model is saved."
        saver.save(sess, _model_path)
        print "Random training is finished."


    def _gen_sentence(self, sess, keyword, p_words, int2word, word2int, rhyme_checker):
        k_lst = [encode_keyword(word2int, keyword)]
        c_lst = [encode_words(word2int, p_words)]
        k_mat = fill_np_matrix(k_lst, _BATCH_SIZE, VOCAB_SIZE-1)
        c_mat = fill_np_matrix(c_lst, _BATCH_SIZE, VOCAB_SIZE-1)
        k_len = fill_np_array(map(len, k_lst), _BATCH_SIZE, 0)
        c_len = fill_np_array(map(len, c_lst), _BATCH_SIZE, 0)
        attention_states = self._get_attention_states(sess, k_mat, c_mat, k_len, c_len)
        s_inputs = sess.run(tf.zeros([_BATCH_SIZE], tf.int32))
        s_state = sess.run(self.s_cell.zero_state(_BATCH_SIZE, tf.float32))
        sentence = ''
        idx = 0
        while idx < rhyme_checker.n_chars:
            s_state, _logits = sess.run([self.s_final_state, self.logits], feed_dict = {
                self.s_inputs: s_inputs,
                self.s_init_state: s_state,
                self.attention_states: attention_states})
            word = decode_word(int2word, rhyme_checker, p_words[-1], _logits.tolist()[0])
            sentence += word
            p_words.append(word)
            _mat = fill_np_matrix([encode_keyword(word2int, word)], _BATCH_SIZE, VOCAB_SIZE-1)
            for i in range(len(word)-1):
                s_state, _ = sess.run([self.s_final_state, self.logits], feed_dict = {
                    self.s_inputs: _mat[:,i].reshape([-1]),
                    self.s_init_state: s_state,
                    self.attention_states: attention_states})
            s_inputs = _mat[:,-1].reshape([-1])
            idx += len(word)
        return sentence

    def generate(self, keywords):
        int2word, word2int = get_vocab()
        rhyme_checker = RhymeChecker()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if not ckpt or not ckpt.model_checkpoint_path:
                self.train(sess)
            _ = Generator._get_saver(sess)
            p_words = [u'^']
            for keyword in keywords:
                sentence = self._gen_sentence(sess, keyword, p_words,
                        int2word, word2int, rhyme_checker)
                uprintln(sentence)
            

if __name__ == '__main__':
    generator = Generator()
    kw_train_data = get_kw_train_data()
    for row in kw_train_data:
        uprintln(row[1:])
        generator.generate(row[1:])

