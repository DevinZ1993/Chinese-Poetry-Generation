#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char2vec import Char2Vec
from char_dict import CharDict
from data_utils import batch_train_data
from paths import save_dir
from random import random
from singleton import Singleton
from utils import CHAR_VEC_DIM, NUM_OF_SENTENCES
import numpy as np
import os
import sys
import tensorflow as tf


_BATCH_SIZE = 128
_NUM_UNITS = 512

_model_path = os.path.join(save_dir, 'model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(Singleton):

    def _encode_keyword(self):
        """ Encode keyword into a vector."""
        self.keyword = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "keyword")
        self.keyword_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "keyword_length")
        _, kw_enc_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.keyword,
                sequence_length = self.keyword_length,
                dtype = tf.float32, 
                time_major = False,
                scope = "keyword_encoder")
        keyword_state = tf.stack(
                values = [tf.concat(kw_enc_states, axis = 1)], 
                axis = 1)
        tf.TensorShape([_BATCH_SIZE, 1, _NUM_UNITS]).\
                assert_same_rank(keyword_state.shape)
        return keyword_state

    def _encode_context(self):
        """ Encode context into a list of vectors. """
        self.context = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "context")
        self.context_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "context_length")
        context_outputs, _ = tf.nn.dynamic_rnn(
                cell = tf.contrib.rnn.GRUCell(_NUM_UNITS),
                inputs = self.context,
                sequence_length = self.context_length,
                dtype = tf.float32, 
                time_major = False,
                scope = "context_encoder")
        tf.TensorShape([_BATCH_SIZE, None, _NUM_UNITS]).\
                assert_same_rank(context_outputs.shape)
        return context_outputs

    def _build_attention(self, keyword_state, context_outputs):
        """ Concatenate keyword and context vectors to build attention. """
        encoder_outputs = tf.concat(
                values = [keyword_state, context_outputs], 
                axis = 1)
        encoder_output_length = tf.add(self.context_length,
                tf.ones(shape = [_BATCH_SIZE], dtype = tf.int32))
        attention = tf.contrib.seq2seq.BahdanauAttention(
                num_units = _NUM_UNITS, 
                memory = encoder_outputs,
                memory_sequence_length = encoder_output_length)
        return attention

    def _decode(self, attention):
        """ Decode attention and reshape into [?, _NUM_UNITS]. """
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.contrib.rnn.GRUCell(_NUM_UNITS),
                attention_mechanism = attention)
        self.decoder_state = decoder_cell.zero_state(
                batch_size = _BATCH_SIZE, dtype = tf.float32)
        self.decoder_inputs = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "decoder_inputs")
        self.decoder_input_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "decoder_input_length")
        decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = decoder_cell,
                inputs = self.decoder_inputs,
                sequence_length = self.decoder_input_length,
                initial_state = self.decoder_state,
                dtype = tf.float32, 
                time_major = False,
                scope = "decoder")
        tf.TensorShape([_BATCH_SIZE, None, _NUM_UNITS]).\
                assert_same_rank(decoder_outputs.shape)
        return self._reshape_decoder_outputs(decoder_outputs)

    def _reshape_decoder_outputs(self, decoder_outputs):
        """ Reshape decoder_outputs into shape [?, _NUM_UNITS]. """
        def concat_output_slices(idx, val):
            output_slice = tf.slice(
                    input_ = decoder_outputs,
                    begin = [idx, 0, 0],
                    size = [1, self.decoder_input_length[idx],  _NUM_UNITS])
            return tf.add(idx, 1),\
                    tf.concat([val, tf.squeeze(output_slice, axis = 0)], 
                            axis = 0)
        tf_i = tf.constant(0)
        tf_v = tf.zeros(shape = [0, _NUM_UNITS], dtype = tf.float32)
        _, reshaped_outputs = tf.while_loop(
                cond = lambda i, v: i < _BATCH_SIZE,
                body = concat_output_slices,
                loop_vars = [tf_i, tf_v],
                shape_invariants = [tf.TensorShape([]),
                    tf.TensorShape([None, _NUM_UNITS])])
        tf.TensorShape([None, _NUM_UNITS]).\
                assert_same_rank(reshaped_outputs.shape)
        return reshaped_outputs

    def _calculate_logits(self, reshaped_outputs):
        """ Calculate logits of decoder outputs. """
        softmax_w = tf.Variable(
                tf.random_normal(shape = [_NUM_UNITS, len(self.char_dict)],
                    mean = 0.0, stddev = 0.08), 
                trainable = True)
        softmax_b = tf.Variable(
                tf.random_normal(shape = [len(self.char_dict)],
                    mean = 0.0, stddev = 0.08),
                trainable = True)
        logits = tf.nn.bias_add(
                tf.matmul(reshaped_outputs, softmax_w),
                bias = softmax_b)
        return logits

    def _minimize_loss(self, logits):
        """ Calculate loss and minimize it. """
        self.targets = tf.placeholder(
                shape = [None],
                dtype = tf.int32, 
                name = "targets")
        labels = tf.one_hot(self.targets, depth = len(self.char_dict))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits = logits,
                labels = labels)
        self.loss = tf.reduce_mean(loss)

        self.global_step = tf.placeholder(
                dtype = tf.int32, name = "global_step")
        learning_rate = tf.train.exponential_decay(
                learning_rate = 0.05, 
                global_step = self.global_step,
                decay_steps = 1,
                decay_rate = 0.97,
                staircase = True)
        self.opt_op = tf.train.AdamOptimizer(learning_rate).\
                minimize(self.loss)

    def _construct_graph(self):
        keyword_state = self._encode_keyword()
        context_outputs = self._encode_context()

        attention = self._build_attention(keyword_state, context_outputs)
        decoder_outputs = self._decode(attention)

        logits = self._calculate_logits(decoder_outputs)
        self.probs = tf.nn.softmax(logits)

        self._minimize_loss(logits)

    def __init__(self):
        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self._construct_graph()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
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
        context = ''
        with tf.Session() as session:
            self._initialize_session(session)
            if not self.trained:
                print("Please train the model first! (./train.py -g)")
                sys.exit(1)
            for keyword in keywords:
                keyword_data, keyword_length = self._fill_np_matrix([keyword])
                context_data, context_length = self._fill_np_matrix([context])
                char = '^'
                while True:
                    decoder_input, decoder_input_length = \
                            self._fill_np_matrix([char])
                    encoder_feed_dict = {
                            self.keyword : keyword_data,
                            self.keyword_length : keyword_length,
                            self.context : context_data,
                            self.context_length : context_length,
                            self.decoder_inputs : decoder_input,
                            self.decoder_input_length : decoder_input_length
                            }
                    if char != '^':
                        encoder_feed_dict[self.decoder_state] = state
                    probs, state = session.run(
                            [self.probs, self.decoder_final_state], 
                            feed_dict = encoder_feed_dict)
                    prob_sums = np.cumsum(probs.tolist()[0])
                    for i, prob_sum in enumerate(prob_sums):
                        if random() < prob_sum:
                            char = self.char_dict.int2char(i)
                            break
                    print(char)
                    if char == '$':
                        break
        return context.split('$')

    def train(self, n_epochs = 6):
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
                        self._train_a_batch(session, epoch,
                                keywords, contexts, sentences)
                        batch_no += 1
                        if 0 == batch_no % 32:
                            self.saver.save(session, _model_path)
                    self.saver.save(session, _model_path)
                print("Training is done.")
            except KeyboardInterrupt:
                print("Training is interrupted.")

    def _train_a_batch(self, session, epoch, keywords, contexts, sentences):
        keyword_data, keyword_length = self._fill_np_matrix(keywords)
        context_data, context_length = self._fill_np_matrix(contexts)
        decoder_inputs, decoder_input_length  = self._fill_np_matrix(
                ['^' + sentence[:-1] for sentence in sentences])
        targets = self._fill_targets(sentences)
        feed_dict = {
                self.global_step : epoch,
                self.keyword : keyword_data,
                self.keyword_length : keyword_length,
                self.context : context_data,
                self.context_length : context_length,
                self.decoder_inputs : decoder_inputs,
                self.decoder_input_length : decoder_input_length,
                self.targets : targets
                }
        loss, _ = session.run([self.loss, self.opt_op],
                feed_dict = feed_dict)
        print(" loss =  %f" % loss)

    def _fill_np_matrix(self, texts):
        max_time = max(map(len, texts))
        matrix = np.zeros([_BATCH_SIZE, max_time, CHAR_VEC_DIM], 
                dtype = np.int32)
        for i in range(_BATCH_SIZE):
            for j in range(max_time):
                matrix[i, j, :] = self.char2vec.get_vect('$')
        for i, text in enumerate(texts):
            matrix[i, : len(text)] = self.char2vec.get_vects(text)
        seq_length = [len(texts[i]) if i < len(texts) else 0 \
                for i in range(_BATCH_SIZE)]
        return matrix, seq_length

    def _fill_targets(self, sentences):
        targets = []
        for sentence in sentences:
            targets.extend(map(self.char_dict.char2int, sentence))
        return targets


# For testing purpose.
if __name__ == '__main__':
    generator = Generator()
    keywords = ['四时', '变', '雪', '新']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)
