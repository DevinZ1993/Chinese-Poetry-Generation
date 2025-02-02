#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import math
import os
import random
import re
import statistics
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
from typing import Any, Iterator

import corpus
import vocab

EMBEDDING_DIM: int = vocab.EMBEDDING_DIM
MODEL_DIM: int = 512

_MODEL_PATH: str = os.path.join(corpus.GENDATA_PATH, 'discriminator.pt')
_NUM_RNN_LAYERS: int = 3
_NUM_EPOCHS: int = 2
_BATCH_SIZE: int = 64
_NUM_BATCHES_FOR_LOGGING: int = 10
_LEARNING_RATE: float = 0.002
_FORCE_TRAINING: bool = True
_MAX_SENTENCE_LEN: int = 11
_SEED: int = 10086


def _gen_code_start_training_data(
        vocab_dict: vocab.Vocab) -> Iterator[tuple[str, bool]]:
    weights: list[float] = [
        float(vocab_dict.get_count_at_index(i)) for i in range(len(vocab_dict))
    ]
    for i in range(1, len(vocab_dict)):
        weights[i] += weights[i - 1]
    all_poems = list(
        corpus.get_poems(lambda ch: ch in vocab_dict, random_order=True))
    print('{} batches'.format(math.ceil(len(all_poems) / _BATCH_SIZE)))
    for poem in all_poems:
        yield ''.join(
            sentence + vocab.END_OF_SENTENCE for sentence in poem), True
        fake_sentences: list[str] = [sentence[0] for sentence in poem]
        for idx in range(len(fake_sentences)):
            sentence_length = random.randint(1, _MAX_SENTENCE_LEN)
            for _ in range(1, sentence_length):
                rand_val = random.uniform(0.0, weights[-1])
                min_idx = 0
                max_idx = len(weights) - 1
                while min_idx < max_idx:
                    mid_idx = min_idx + (max_idx - min_idx) // 2
                    if weights[mid_idx] >= rand_val:
                        max_idx = mid_idx
                    else:
                        min_idx = mid_idx + 1
                fake_sentences[idx] += vocab_dict[min_idx]
            fake_sentences[idx] += vocab.END_OF_SENTENCE
        yield ''.join(fake_sentences), False


def _get_learning_rate(step: float) -> float:
    return _LEARNING_RATE


class Discriminator(nn.Module):

    def __init__(self, vocab_instance: vocab.Vocab):
        super(Discriminator, self).__init__()
        self.vocab = vocab_instance
        self.vocab_size = len(self.vocab)
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=MODEL_DIM,
                            num_layers=_NUM_RNN_LAYERS,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(2 * MODEL_DIM, 1)
        self.training_epoch: int = 0
        self.optimizer = optim.Adam(self.parameters(), lr=_LEARNING_RATE)
        self.lr_sched = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_get_learning_rate)
        self._prepare_model()

    def load_model(self) -> None:
        checkpoint = torch.load(_MODEL_PATH)
        self.training_epoch = checkpoint.get('epoch', 0)
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_sched.load_state_dict(checkpoint['lr_sched'])

    def save_model(self) -> None:
        checkpoint['epoch'] = self.training_epoch
        checkpoint['model'] = self.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['lr_sched'] = self.lr_sched.state_dict()
        checkpoint_path = f'{_MODEL_PATH}.tmp'
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint {checkpoint_path}.')
        os.replace(checkpoint_path, _MODEL_PATH)

    def grad_norm(self) -> float:
        total_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item()**2
        return math.sqrt(total_norm)

    def _prepare_model(self) -> None:
        if not _FORCE_TRAINING and os.path.exists(_MODEL_PATH):
            self.load_model()
        if self.training_epoch < _NUM_EPOCHS:
            self.train()
            while self.training_epoch < _NUM_EPOCHS:
                self._train_on_single_epoch(
                    _gen_code_start_training_data(self.vocab))
                self.training_epoch += 1
                self.save_model()
        self.eval()

    def _train_on_single_epoch(self, data: Iterator[tuple[str, bool]]) -> None:
        seq_buffer: list[str] = []
        label_buffer: list[bool] = []
        loss_vals: list[float] = []
        grad_norm_sum: float = 0.0
        print(f'Epoch #{self.training_epoch}')
        for seq, label in data:
            seq_buffer.append(seq)
            label_buffer.append(label)
            if len(seq_buffer) == _BATCH_SIZE:
                loss_vals.append(
                    self._train_on_single_batch(seq_buffer, label_buffer))
                grad_norm_sum += self.grad_norm()
                if len(loss_vals) % _NUM_BATCHES_FOR_LOGGING == 0:
                    grad_norm = grad_norm_sum / _NUM_BATCHES_FOR_LOGGING
                    grad_norm_sum = 0.0
                    print('Epoch #{}, {} batches: loss = {} grad_norm = {}'.
                          format(
                              self.training_epoch, len(loss_vals),
                              statistics.mean(
                                  loss_vals[-_NUM_BATCHES_FOR_LOGGING:]),
                              grad_norm))
                seq_buffer = []
                label_buffer = []
        if seq_buffer:
            loss_vals.append(
                self._train_on_single_batch(seq_buffer, label_buffer))
        print(
            'Finished epoch #{} ({} batches in total): loss = {} (stdev = {})'.
            format(self.training_epoch, len(loss_vals),
                   statistics.mean(loss_vals), statistics.stdev(loss_vals)))
        self.lr_sched.step()

    def _train_on_single_batch(self, sequences: list[str],
                               labels: list[bool]) -> float:
        self.optimizer.zero_grad()
        logits = self(sequences)
        targets = torch.tensor(labels, dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, sequences: list[str]) -> torch.Tensor:
        batch_size = len(sequences)
        seq_lengths: list[int] = [len(seq) for seq in sequences]
        max_seq_length: int = max(seq_lengths)
        input_tensor = torch.zeros(batch_size, max_seq_length,
                                   self.vocab.embedding_dim)
        for seq_idx, seq in enumerate(sequences):
            input_tensor[seq_idx, :len(seq)] = self.vocab.embedding(seq)
        input_tensor = self.dropout(input_tensor)
        hidden_state = torch.zeros(2 * _NUM_RNN_LAYERS, batch_size, MODEL_DIM)
        cell_state = torch.zeros(2 * _NUM_RNN_LAYERS, batch_size, MODEL_DIM)
        lstm_input = nn.utils.rnn.pack_padded_sequence(input_tensor,
                                                       seq_lengths,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        _, (hidden_state, _) = self.lstm(lstm_input, (hidden_state, cell_state))
        logits = self.linear(hidden_state[-2:, :, :].permute(1, 0, 2).reshape(
            batch_size, -1))  # B x 1
        return logits.view(-1)

    def evaluate(self, sequences: list[str]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(sequences)
            probs = F.sigmoid(logits)
        return probs


def _global_init():
    random.seed(_SEED)
    torch.manual_seed(_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)


if __name__ == '__main__':
    _global_init()
    vocab_dict = vocab.Vocab(vocab.EMBEDDING_DIM)
    discriminator = Discriminator(vocab_dict)
