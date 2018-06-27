#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char2vec import Char2Vec
from char_dict import CharDict, end_of_sentence, start_of_sentence
from data_utils import batch_train_data
from paths import save_dir
from pron_dict import PronDict
from random import random
from singleton import Singleton
from utils import CHAR_VEC_DIM, NUM_OF_SENTENCES
import numpy as np
import os
import sys
import tensorflow as tf


_BATCH_SIZE = 64
_NUM_UNITS = 512

_model_path = os.path.join(save_dir, 'model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(Singleton):

    def _build_keyword_encoder(self):
        """ Encode keyword into a vector."""
        self.keyword = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "keyword")
        self.keyword_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "keyword_length")
        _, bi_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.keyword,
                sequence_length = self.keyword_length,
                dtype = tf.float32, 
                time_major = False,
                scope = "keyword_encoder")
        self.keyword_state = tf.concat(bi_states, axis = 1)
        tf.TensorShape([_BATCH_SIZE, _NUM_UNITS]).\
                assert_same_rank(self.keyword_state.shape)

    def _build_context_encoder(self):
        """ Encode context into a list of vectors. """
        self.context = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "context")
        self.context_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "context_length")
        bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.context,
                sequence_length = self.context_length,
                dtype = tf.float32, 
                time_major = False,
                scope = "context_encoder")
        self.context_outputs = tf.concat(bi_outputs, axis = 2)
        tf.TensorShape([_BATCH_SIZE, None, _NUM_UNITS]).\
                assert_same_rank(self.context_outputs.shape)

    def _build_decoder(self):
        """ Decode keyword and context into a sequence of vectors. """
        attention = tf.contrib.seq2seq.BahdanauAttention(
                num_units = _NUM_UNITS, 
                memory = self.context_outputs,
                memory_sequence_length = self.context_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.contrib.rnn.GRUCell(_NUM_UNITS),
                attention_mechanism = attention)
        self.decoder_init_state = decoder_cell.zero_state(
                batch_size = _BATCH_SIZE, dtype = tf.float32).\
                        clone(cell_state = self.keyword_state)
        self.decoder_inputs = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "decoder_inputs")
        self.decoder_input_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "decoder_input_length")
        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = decoder_cell,
                inputs = self.decoder_inputs,
                sequence_length = self.decoder_input_length,
                initial_state = self.decoder_init_state,
                dtype = tf.float32, 
                time_major = False,
                scope = "training_decoder")
        tf.TensorShape([_BATCH_SIZE, None, _NUM_UNITS]).\
                assert_same_rank(self.decoder_outputs.shape)

    def _build_projector(self):
        """ Project decoder_outputs into character space. """
        softmax_w = tf.Variable(
                tf.random_normal(shape = [_NUM_UNITS, len(self.char_dict)],
                    mean = 0.0, stddev = 0.08), 
                trainable = True)
        softmax_b = tf.Variable(
                tf.random_normal(shape = [len(self.char_dict)],
                    mean = 0.0, stddev = 0.08),
                trainable = True)
        reshaped_outputs = self._reshape_decoder_outputs()
        self.logits = tf.nn.bias_add(
                tf.matmul(reshaped_outputs, softmax_w),
                bias = softmax_b)
        self.probs = tf.nn.softmax(self.logits)

    def _reshape_decoder_outputs(self):
        """ Reshape decoder_outputs into shape [?, _NUM_UNITS]. """
        def concat_output_slices(idx, val):
            output_slice = tf.slice(
                    input_ = self.decoder_outputs,
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

    def _build_optimizer(self):
        """ Define cross-entropy loss and minimize it. """
        self.targets = tf.placeholder(
                shape = [None],
                dtype = tf.int32, 
                name = "targets")
        labels = tf.one_hot(self.targets, depth = len(self.char_dict))
        cross_entropy = tf.losses.softmax_cross_entropy(
                onehot_labels = labels,
                logits = self.logits)
        self.loss = tf.reduce_mean(cross_entropy)

        self.learning_rate = tf.clip_by_value(
                tf.multiply(1.6e-5, tf.pow(2.1, self.loss)),
                clip_value_min = 0.0002,
                clip_value_max = 0.02)
        self.opt_step = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate).\
                        minimize(loss = self.loss)

    def _build_graph(self):
        self._build_keyword_encoder()
        self._build_context_encoder()
        self._build_decoder()
        self._build_projector()
        self._build_optimizer()

    def __init__(self):
        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self._build_graph()
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
        pron_dict = PronDict()
        context = start_of_sentence()
        with tf.Session() as session:
            self._initialize_session(session)
            if not self.trained:
                print("Please train the model first! (./train.py -g)")
                sys.exit(1)
            for keyword in keywords:
                keyword_data, keyword_length = self._fill_np_matrix(
                        [keyword] * _BATCH_SIZE)
                context_data, context_length = self._fill_np_matrix(
                        [context] * _BATCH_SIZE)
                char = start_of_sentence()
                for _ in range(7):
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
                    if char == start_of_sentence():
                        pass
                    else:
                        encoder_feed_dict[self.decoder_init_state] = state
                    probs, state = session.run(
                            [self.probs, self.decoder_final_state], 
                            feed_dict = encoder_feed_dict)
                    prob_list = self._gen_prob_list(probs, context, pron_dict)
                    prob_sums = np.cumsum(prob_list)
                    rand_val = prob_sums[-1] * random()
                    for i, prob_sum in enumerate(prob_sums):
                        if rand_val < prob_sum:
                            char = self.char_dict.int2char(i)
                            break
                    context += char
                context += end_of_sentence()
        return context[1:].split(end_of_sentence())

    def _gen_prob_list(self, probs, context, pron_dict):
        prob_list = probs.tolist()[0]
        prob_list[0] = 0
        prob_list[-1] = 0
        idx = len(context)
        used_chars = set(ch for ch in context)
        for i in range(1, len(prob_list) - 1):
            ch = self.char_dict.int2char(i)
            # Penalize used characters.
            if ch in used_chars:
                prob_list[i] *= 0.6
            # Penalize rhyming violations.
            if (idx == 15 or idx == 31) and \
                    not pron_dict.co_rhyme(ch, context[7]):
                prob_list[i] *= 0.2
            # Penalize tonal violations.
            if idx > 2 and 2 == idx % 8 and \
                    not pron_dict.counter_tone(context[2], ch):
                prob_list[i] *= 0.4
            if (4 == idx % 8 or 6 == idx % 8) and \
                    not pron_dict.counter_tone(context[idx - 2], ch):
                prob_list[i] *= 0.4
        return prob_list

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
                [start_of_sentence() + sentence[:-1] \
                        for sentence in sentences])
        targets = self._fill_targets(sentences)
        feed_dict = {
                self.keyword : keyword_data,
                self.keyword_length : keyword_length,
                self.context : context_data,
                self.context_length : context_length,
                self.decoder_inputs : decoder_inputs,
                self.decoder_input_length : decoder_input_length,
                self.targets : targets
                }
        loss, learning_rate, _ = session.run(
                [self.loss, self.learning_rate, self.opt_step],
                feed_dict = feed_dict)
        print(" loss =  %f, learning_rate = %f" % (loss, learning_rate))

    def _fill_np_matrix(self, texts):
        max_time = max(map(len, texts))
        matrix = np.zeros([_BATCH_SIZE, max_time, CHAR_VEC_DIM], 
                dtype = np.int32)
        for i in range(_BATCH_SIZE):
            for j in range(max_time):
                matrix[i, j, :] = self.char2vec.get_vect(end_of_sentence())
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

