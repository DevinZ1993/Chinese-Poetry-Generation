#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import *
from rhyme import RhymeDict
from word2vec import get_word_embedding
from data_utils import *
from collections import deque
import tensorflow as tf
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_model_path = os.path.join(save_dir, 'model')

_NUM_UNITS = 128
_NUM_LAYERS = 4
_BATCH_SIZE = 64


class Generator:

    def __init__(self):
        embedding = tf.Variable(tf.constant(0.0, shape=[VOCAB_SIZE, _NUM_UNITS]), trainable = False)
        self._embed_ph = tf.placeholder(tf.float32, [VOCAB_SIZE, _NUM_UNITS])
        self._embed_init = embedding.assign(self._embed_ph)

        self.encoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
        self.encoder_init_state = self.encoder_cell.zero_state(_BATCH_SIZE, dtype = tf.float32)
        self.encoder_inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.encoder_lengths = tf.placeholder(tf.int32, [_BATCH_SIZE])
        _, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell = self.encoder_cell,
                initial_state = self.encoder_init_state,
                inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs),
                sequence_length = self.encoder_lengths,
                scope = 'encoder')

        self.decoder_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(_NUM_UNITS)] * _NUM_LAYERS)
        self.decoder_init_state = self.encoder_cell.zero_state(_BATCH_SIZE, dtype = tf.float32)
        self.decoder_inputs = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        self.decoder_lengths = tf.placeholder(tf.int32, [_BATCH_SIZE])
        outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = self.decoder_cell,
                initial_state = self.decoder_init_state,
                inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs),
                sequence_length = self.decoder_lengths,
                scope = 'decoder')

        with tf.variable_scope('decoder'):
            softmax_w = tf.get_variable('softmax_w', [_NUM_UNITS, VOCAB_SIZE])
            softmax_b = tf.get_variable('softmax_b', [VOCAB_SIZE])

        logits = tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, _NUM_UNITS]), softmax_w),
                bias = softmax_b)
        self.probs = tf.nn.softmax(logits)

        self.targets = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        labels = tf.one_hot(tf.reshape(self.targets, [-1]), depth = VOCAB_SIZE)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = logits,
                labels = labels)
        self.loss = tf.reduce_mean(loss)

        self.learn_rate = tf.Variable(0.0, trainable = False)
        self.opt_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables())
        self.int2ch, self.ch2int = get_vocab()

    def _init_vars(self, sess):
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
            sess.run([self._embed_init], feed_dict = {
                self._embed_ph: get_word_embedding(_NUM_UNITS)})
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def _train_a_batch(self, sess, kw_mats, kw_lens, s_mats, s_lens):
        total_loss = 0
        for idx in range(4):
            encoder_feed_dict = {self.encoder_inputs: kw_mats[idx],
                    self.encoder_lengths: kw_lens[idx]}
            if idx > 0:
                encoder_feed_dict[self.encoder_init_state] = state
            state = sess.run(self.encoder_final_state,
                    feed_dict = encoder_feed_dict)
            state, loss, _ = sess.run([self.decoder_final_state, self.loss, self.opt_op], feed_dict = {
                self.decoder_init_state: state,
                self.decoder_inputs: s_mats[idx][:,:-1],
                self.decoder_lengths: s_lens[idx],
                self.targets: s_mats[idx][:,1:]})
            total_loss += loss
        print "loss = %f" %(total_loss/4)

    def train(self, n_epochs = 6, learn_rate = 0.002, decay_rate = 0.97):
        print "Start training RNN enc-dec model ..."
        with tf.Session() as sess:
            self._init_vars(sess)
            try:
                for epoch in range(n_epochs):
                    batch_no = 0
                    sess.run(tf.assign(self.learn_rate, learn_rate * decay_rate ** epoch))
                    for kw_mats, kw_lens, s_mats, s_lens in batch_train_data(_BATCH_SIZE):
                        print "[Training Seq2Seq] epoch = %d/%d, line %d to %d ..." \
                                %(epoch, n_epochs, batch_no*_BATCH_SIZE, (batch_no+1)*_BATCH_SIZE),
                        self._train_a_batch(sess, kw_mats, kw_lens, s_mats, s_lens)
                        batch_no += 1
                        if 0 == batch_no%32:
                            self.saver.save(sess, _model_path)
                            print "[Training Seq2Seq] The temporary model has been saved."
                    self.saver.save(sess, _model_path)
                print "Training has finished."
            except KeyboardInterrupt:
                print "\nTraining is interrupted."

    def generate(self, keywords):
        sentences = []
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            self.train(1)
        with tf.Session() as sess:
            self._init_vars(sess)
            rdict = RhymeDict()
            length = -1
            rhyme_ch = None
            for idx, keyword in enumerate(keywords):
                kw_mat = fill_np_matrix([[self.ch2int[ch] for ch in keyword]], _BATCH_SIZE, VOCAB_SIZE-1)
                kw_len = fill_np_array([len(keyword)], _BATCH_SIZE, 0)
                encoder_feed_dict = {self.encoder_inputs: kw_mat,
                        self.encoder_lengths: kw_len}
                if idx > 0:
                    encoder_feed_dict[self.encoder_init_state] = state
                state = sess.run(self.encoder_final_state,
                        feed_dict = encoder_feed_dict)
                sentence = u''
                decoder_inputs = np.zeros([_BATCH_SIZE, 1], dtype = np.int32)
                decoder_lengths = fill_np_array([1], _BATCH_SIZE, 0)
                i = 0
                while True:
                    probs, state = sess.run([self.probs, self.decoder_final_state], feed_dict = {
                        self.decoder_init_state: state,
                        self.decoder_inputs: decoder_inputs,
                        self.decoder_lengths: decoder_lengths})
                    prob_list = probs.tolist()[0]
                    prob_list[0] = 0.
                    if length > 0:
                        if i  == length:
                            prob_list = [.0]*VOCAB_SIZE
                            prob_list[-1] = 1.
                        elif i == length-1:
                            for j, ch in enumerate(self.int2ch):
                                if  0 == j or VOCAB_SIZE-1 == j:
                                    prob_list[j] = 0.
                                else:
                                    rhyme = rdict.get_rhyme(ch)
                                    tone = rdict.get_tone(ch)
                                    if (1 == idx and 'p' != tone) or \
                                            (2 == idx and (rdict.get_rhyme(rhyme_ch) == rhyme or 'z' != tone)) or \
                                            (3 == idx and (ch == rhyme_ch or rdict.get_rhyme(rhyme_ch) != rhyme or 'p' != tone)):
                                        prob_list[j] = 0.
                        else:
                            prob_list[-1] = 0.
                    else:
                        if i != 5 and i != 7:
                            prob_list[-1] = 0.
                    prob_sums = np.cumsum(prob_list)
                    if prob_sums[-1] == 0.:
                        prob_list = probs.tolist()[0]
                        prob_sums = np.cumsum(prob_list)
                    for j in range(VOCAB_SIZE-1, -1, -1):
                        if random.random() < prob_list[j]/prob_sums[j]:
                            ch = self.int2ch[j]
                            break
                    #ch = self.int2ch[np.argmax(prob_list)]
                    if idx == 1 and i == length-1:
                        rhyme_ch = ch
                    if ch == self.int2ch[-1]:
                        length = i
                        break
                    else:
                        sentence += ch
                        decoder_inputs[0,0] = self.ch2int[ch]
                        i += 1
                #uprintln(sentence)
                sentences.append(sentence)
        return sentences


if __name__ == '__main__':
    generator = Generator()
    kw_train_data = get_kw_train_data()
    for row in kw_train_data[100:]:
        uprintln(row)
        generator.generate(row)
        print

