#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from utils import *
from singleton import Singleton
import os
import tensorflow as tf
import sys


_BATCH_SIZE = 128
_NUM_UNITS = 512

_model_path = os.path.join(save_dir, 'model')


class Generator(Singleton):

    def _construct_graph(self):
        # Encode the keyword into a vector with BiGRU.
        self.keyword = tf.placeholder(tf.int32, 
                [_BATCH_SIZE, None, CHAR_VEC_DIM])
        _, kw_enc_states = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.keyword,
                dtype = tf.float32, time_major = False)
        keyword_state = tf.concat(kw_enc_states, axis = 1)

        # Encode the context into a sequence of vectors with BiGRU, 
        #   preceded by the keyword vector.
        self.context = tf.placeholder(tf.int32, 
                [_BATCH_SIZE, None, CHAR_VEC_DIM])
        context_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.context,
                dtype = tf.float32, time_major = False)
        encoder_outputs = tf.concat([keyword_state, context_outputs], axis = 1)

        # LSTM decoder with attention mechanism.
        self.decoder_inputs = tf.placeholder(tf.int32, 
                [_BATCH_SIZE, None, CHAR_VEC_DIM])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(_NUM_UNITS, 
                encoder_outputs)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                tf.contrib.rnn.LSTMCell(_NUM_UNITS),
                attention_mechanism, attention_size = _NUM_UNITS)
        self.decoder_state = decoder_cell.zero_state()
        decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = decoder_cell,
                inputs = self.sentence,
                initial_state = self.decoder_state,
                dtype = tf.float32, time_major = False)

        # Generate output distribution with softmax.
        softmax_w = tf.Variable(
                tf.constant(0.0, shape = [_NUM_UNITS, VOCAB_SIZE]), 
                trainable = True)
        softmax_b = tf.Variable(
                tf.constant(0.0, shape = [VOCAB_SIZE]),
                trainable = True)
        reshaped_outputs = tf.reshape(decoder_outputs, [-1, _NUM_UNITS])
        logits = tf.nn.bias_add(
                tf.matmul(reshaped_outputs, softmax_w),
                bias = softmax_b)
        self.probs = tf.nn.softmax(logits)

        # Define cross-entropy loss.
        self.targets = tf.placeholder(tf.int32, [_BATCH_SIZE, None])
        reshaped_targets = tf.reshape(self.targets, [-1])
        labels = tf.one_hot(reshaped_targets, depth = VOCAB_SIZE)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = logits,
                labels = labels)
        self.loss = tf.reduce_mean(loss)

        # Define learning method.
        self.global_step = tf.placeholder(tf.int32, trainable = False)
        learning_rate = tf.train.exponential_decay(
                learning_rate = 0.002, 
                global_step = global_step,,
                decay_steps = 1,
                decay_rate = 0.97,
                staircase = True)
        self.opt_op = tf.train.AdamOptimizer(learning_rate).\
                minimize(self.loss, global_step = global_step)

    def __init__(self):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self._construct_graph()
        self.trained = False
        
    def _initialize_session(self, session):
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            session.run(init_op)
        else:
            self.saver.restore(session, checkpoint.model_checkpoint_path)
            self.trained = True

    def generate(self, keywords):
        assert NUM_OF_SENTENCES == len(keywords)
        with tf.Session() as session:
            self._initialize_session(session)
            if not self.trained:
                print("Please train the model first! (./train.py -g)")
                sys.exit(1)
            context = ''
            for keyword in keywords:
                char = '^'
                while True:
                    encoder_feed_dict = {
                            self.keyword = self.char2vec.get_vects(keyword),
                            self.context = self.char2vec.get_vects(context),
                            self.decoder_inputs = self.char2vec.get_vec(char),
                            }
                    if char != '^':
                        encoder_feed_dict[self.decoder_state] = state
                    probs, state = session.run(
                            [self.probs, self.decoder_final_state], 
                            feeed_dict = encoder_feed_dict)
                    if np.argmax(probs) == len(self.char_dict) - 1:
                        char = '$'
                        break
                    else:
                        char = self.char_dict.int2char(np.argmax(probs))
                        context += char
                context += '$'

    def train(self):
        print("Training RNN-based generator ...")
        with tf.Session() as session:
            self._initialize_session(session)
            try:
                for epoch in range(n_epochs):
                    batch_no = 0
                    for keywords, contexts, sentences \
                            in batch_train_data(_BATCH_SIZE):
                        sys.stdout.write("[Seq2Seq Training] epoch = %d, " \
                                "line %d to %d ..." % 
                                (epoch, batch_no * _BATCH_SIZE,
                                (batch_no + 1) * _BATCH_SIZE))
                        sys.stdout.flush()
                        self._train_a_batch(session,
                                keywords, contexts, sentences)
                        batch_no += 1
                        if 0 == batch_no % 32:
                            self.saver.save(session, _model_path)
                    self.saver.save(session, _model_path)
                print("Training is done.")
            except KeyboardInterrupt:
                print("Training is interrupted.")

    def _train_a_batch(self, session, keywords, contexts, sentences):
        keyword_batch = self._fill_np_matrix(keywords)
        context_batch = self._fill_np_matrix(contexts)
        decoder_inputs = self._fill_np_matrix(
                map(lambda line: '^' + line[:-1], sentences))
        targets = self._fill_targets(sentences)
        feed_dict = {
                self.keyword = keyword_batch,
                self.context = context_batch,
                self.decoder_inputs = decoder_inputs,
                self.targets = targets
                }
        loss, _ = session.run([self.loss, self.opt_op],
                feed_dict = feed_dict)
        print("loss = " % loss)

    def _fill_np_matrix(self, texts):
        max_time = max(map(len, texts))
        matrix = np.full([_BATCH_SIZE, max_time, CHAR_VEC_DIM], 
                len(self.char_dict) - 1, dtype = np.int32)
        for i, text in enumerate(texts):
            matrix[i, : len(text)] = self.char2vec.get_vects(text)
        return matrix

    def _fill_targest(self, sentences):
        max_time = max(map(len, sentences))
        matrix = np.full([_BATCH_SIZE, max_time], 
                len(self.char_dict) - 1, dtype = np.int32)
        for i, sentence in enumerate(sentences):
            code_list = list(map(self.char_dict.char2int, sentence))
            matrix[i, : len(sentence)] = np.array(code_list)
        return matrix


# For testing purpose.
if __name__ == '__main__':
    generator = Generator()
    keywords = ['春天', '桃花', '燕', '柳']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)
