#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import common
import corpus
import discriminator
import generator
import vocab

_NUM_ACTION_SAMPLES: int = 3
_NUM_ROLLOUTS: int = 3
_PRETRAINING_WEIGHT: float = 0.2
_NUM_RL_EPOCHS: int = 3
_NUM_DISCRIMINATOR_BATCHES: int = 600
_NUM_GENERATOR_BATCHES: int = 4000


def train_seq_gan():
    vocab_dict = vocab.Vocab(vocab.EMBEDDING_DIM)
    g = generator.Generator(vocab_dict)
    d = discriminator.Discriminator(vocab_dict)
    while g.training_epoch < generator.TOTAL_PRETRAINING_EPOCHS + _NUM_RL_EPOCHS:
        d.train_on_all_poems(g,
                             target_epochs=(g.training_epoch + 1 -
                                            generator.TOTAL_PRETRAINING_EPOCHS +
                                            discriminator.COLD_START_EPOCHS),
                             num_batches=_NUM_DISCRIMINATOR_BATCHES,
                             poem_filter=lambda p: corpus.is_qiyanjueju(p) or
                             corpus.is_qiyanlvshi(p))
        g.train_rl_model(num_batches=_NUM_GENERATOR_BATCHES,
                         k_actions=_NUM_ACTION_SAMPLES,
                         m_rollouts=_NUM_ROLLOUTS,
                         pretraining_weight=_PRETRAINING_WEIGHT,
                         discriminator=d)


if __name__ == '__main__':
    common.global_init()
    train_seq_gan()
