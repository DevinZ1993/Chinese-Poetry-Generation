#! /usr/bin/env python3
# -*- coding:utf-8 -*-
"""Vocabulary words and their context-free embeddings."""

import bisect
from dataclasses import dataclass
import heapq
import math
import os
import random
import re
import statistics
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Iterator

import common
import corpus

EMBEDDING_DIM: int = 256
END_OF_SENTENCE: str = '$'

_VOCAB_PATH: str = os.path.join(corpus.GENDATA_PATH, 'vocab.txt')
_EMBEDDING_PATH: str = os.path.join(corpus.GENDATA_PATH, 'embeddings.pt')
_MIN_WORD_COUNT: int = 30
_MAX_WINDOW_SIZE: int = 7
_NUM_NOISE_WORDS: int = 5
_NUM_EPOCHS_FOR_WORDS: int = 30
_NUM_EPOCHS_FOR_SENTENCES: int = 5
_INIT_LR_FOR_WORDS: float = 0.002
_INIT_LR_FOR_SENTENCES: float = 0.001
_LR_DECAY_FOR_WORDS: float = 0.9
_LR_DECAY_FOR_SENTENCES: float = 0.3
_BATCH_SIZE: int = 64
_NUM_BATCHES_FOR_LOGGING: int = 5000
_FORCE_TRAINING: bool = False


class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size: int, dim: int):
        super(EmbeddingModel, self).__init__()
        self.u_embeds = nn.Embedding(vocab_size, dim)
        self.v_embeds = nn.Embedding(vocab_size, dim)
        max_init_weight: float = 1.0 / math.sqrt(dim)
        nn.init.uniform_(self.u_embeds.weight,
                         a=-max_init_weight,
                         b=max_init_weight)
        nn.init.uniform_(self.v_embeds.weight,
                         a=-max_init_weight,
                         b=max_init_weight)

    def forward(self, centers: torch.Tensor,
                context_and_negatives: torch.Tensor) -> torch.Tensor:
        v_tensors = self.v_embeds(centers)
        u_tensors = self.u_embeds(context_and_negatives)
        dot_product = torch.matmul(v_tensors.unsqueeze(1),
                                   u_tensors.transpose(1, 2)).squeeze(1)
        return F.sigmoid(dot_product)

    def grad_norm(self) -> float:
        total_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item()**2
        return math.sqrt(total_norm)


def _calculate_batch_mean_loss(model: EmbeddingModel, centers: list[int],
                               context_and_negatives: list[list[int]]) -> float:
    logits = model(torch.LongTensor(centers),
                   torch.LongTensor(context_and_negatives))
    targets = torch.zeros_like(logits)
    targets[:, 0] = 1
    loss_function = torch.nn.BCELoss()
    loss = loss_function(logits, targets)
    loss.backward()
    return loss.item()


class EmbeddingTrainer:
    """A helper class for training the skip-gram embedding model.

    The training happens in two phases:
    1. In the first phase, train the model on single words from the
    ShiXueHanYing dictionary.
    2. In the second phase, train the model on poem sentences from a larger
    corpus.
    """

    vocab_size: int
    embedding_dim: int
    _indices: dict[str, int]
    _counts: list[int]
    _training_epoch: int

    def __init__(self, embedding_dim: int, indices: dict[str, int],
                 counts: list[int]):
        assert len(indices) == len(counts)
        self.vocab_size = len(indices)
        self.embedding_dim = embedding_dim
        self._indices = indices
        self._counts = counts
        self._training_epoch = 0
        self._model = EmbeddingModel(self.vocab_size, self.embedding_dim)
        self._optimizer = None
        self._lr_sched = None

    def gen_embeddings(self) -> nn.Embedding:
        if not _FORCE_TRAINING and os.path.exists(_EMBEDDING_PATH):
            self._load_embedding_model()
        if self._training_epoch < (_NUM_EPOCHS_FOR_WORDS +
                                   _NUM_EPOCHS_FOR_SENTENCES):
            self._train_embedding_model()
        self._model.eval()
        return self._model.v_embeds

    def _load_embedding_model(self):
        checkpoint = torch.load(_EMBEDDING_PATH, weights_only=True)
        self._training_epoch = checkpoint.get('epoch', 0)
        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=_INIT_LR_FOR_WORDS if self._training_epoch
            < _NUM_EPOCHS_FOR_WORDS else _INIT_LR_FOR_SENTENCES)
        optimizer_state_dict = checkpoint.get('optimizer')
        if optimizer_state_dict:
            self._optimizer.load_state_dict(optimizer_state_dict)
        self._lr_sched = optim.lr_scheduler.StepLR(
            self._optimizer,
            step_size=1,
            gamma=_LR_DECAY_FOR_WORDS if self._training_epoch
            < _NUM_EPOCHS_FOR_WORDS else _LR_DECAY_FOR_SENTENCES)
        lr_sched_state_dict = checkpoint.get('lr_sched')
        if lr_sched_state_dict:
            self._lr_sched.load_state_dict(lr_sched_state_dict)
        self._model.load_state_dict(checkpoint['model'])
        embedding_shape = self._model.v_embeds.weight.shape
        if embedding_shape[0] != self.vocab_size:
            raise Exception(f'Expected {self.vocab_size} embeddings but got'
                            f' {embedding_shape[0]}.')
        if embedding_shape[1] != self.embedding_dim:
            raise Exception(
                f'Expected {self.embedding_dim}-dim embeddings but the actual'
                f' dimension is {embedding_shape[1]}')

    def _train_embedding_model(self) -> None:
        """Trains a skip-gram model with negative sampling."""

        print('Training token embeddings ...')
        self._model.train()
        while self._training_epoch < (_NUM_EPOCHS_FOR_WORDS +
                                      _NUM_EPOCHS_FOR_SENTENCES):
            print(f'Epoch #{self._training_epoch} ...')

            if (not self._optimizer or not self._lr_sched or
                    self._training_epoch == _NUM_EPOCHS_FOR_WORDS):
                # Switch to a different optimization policy on phase 2.
                self._optimizer = optim.Adam(
                    self._model.parameters(),
                    lr=_INIT_LR_FOR_WORDS if self._training_epoch
                    < _NUM_EPOCHS_FOR_WORDS else _INIT_LR_FOR_SENTENCES)
                self._lr_sched = optim.lr_scheduler.StepLR(
                    self._optimizer,
                    step_size=1,
                    gamma=_LR_DECAY_FOR_WORDS if self._training_epoch
                    < _NUM_EPOCHS_FOR_WORDS else _LR_DECAY_FOR_SENTENCES)

            loss_vals = []
            grad_norm_sum = 0.0
            for centers, context_and_negatives in self._get_batch_data():
                assert len(centers) == len(context_and_negatives)
                self._optimizer.zero_grad()
                loss_vals.append(
                    _calculate_batch_mean_loss(self._model, centers,
                                               context_and_negatives))
                grad_norm_sum += self._model.grad_norm()
                self._optimizer.step()
                if len(loss_vals) % _NUM_BATCHES_FOR_LOGGING == 0:
                    loss_val = statistics.mean(
                        loss_vals[-_NUM_BATCHES_FOR_LOGGING:])
                    grad_norm = grad_norm_sum / _NUM_BATCHES_FOR_LOGGING
                    grad_norm_sum = 0.0
                    print(
                        'Epoch #{} ({} batches done): loss = {} grad_norm = {}'.
                        format(self._training_epoch, len(loss_vals), loss_val,
                               grad_norm))
            print(
                'Finished epoch #{} ({} batches in total): loss = {} (stdev={})'
                .format(self._training_epoch, len(loss_vals),
                        statistics.mean(loss_vals),
                        statistics.stdev(loss_vals)))
            self._lr_sched.step()
            self._training_epoch += 1
            if self._training_epoch < _NUM_EPOCHS_FOR_WORDS:
                # Skip saving incremental checkpoints for phase 1.
                continue

            checkpoint_path = f'{_EMBEDDING_PATH}.tmp'
            checkpoint = {
                'epoch': self._training_epoch,
                'optimizer': self._optimizer.state_dict(),
                'lr_sched': self._lr_sched.state_dict(),
                'model': self._model.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            os.replace(checkpoint_path, _EMBEDDING_PATH)
            print(f'Saved checkpoint in {_EMBEDDING_PATH}.')
        print('Finished training token embeddings.')

    def _get_batch_data(self) -> Iterator[tuple[list[int], list[list[int]]]]:
        center_batch: list[int] = []
        context_and_negative_batch: list[list[int]] = []
        for center, context_and_negatives in self._get_single_datum():
            center_batch.append(center)
            context_and_negative_batch.append(context_and_negatives)
            if len(center_batch) == _BATCH_SIZE:
                yield center_batch, context_and_negative_batch
                center_batch = []
                context_and_negative_batch = []
        if center_batch:
            yield center_batch, context_and_negative_batch

    def _get_single_datum(self) -> Iterator[tuple[int, list[int]]]:
        sample_weights = []
        weight_sum = 0.0
        for count in self._counts[:-1]:
            weight_sum += count**0.75
            sample_weights.append(weight_sum)
        for sentence in self._gen_training_sentences():
            n = len(sentence)
            for i in range(n):
                left_idx = self._indices.get(sentence[i])
                if left_idx is None:
                    continue
                for j in range(i + 1, min(n, i + _MAX_WINDOW_SIZE)):
                    right_idx = self._indices.get(sentence[j])
                    if right_idx is None:
                        continue
                    # Consider the left word as the center.
                    context_and_negatives = [right_idx]
                    for _ in range(_NUM_NOISE_WORDS):
                        rand_num = random.uniform(0.0, sample_weights[-1])
                        idx = bisect.bisect_left(sample_weights, rand_num)
                        context_and_negatives.append(idx)
                    yield left_idx, context_and_negatives
                    # Consider the right word as the center.
                    context_and_negatives = [left_idx]
                    for _ in range(_NUM_NOISE_WORDS):
                        rand_num = random.uniform(0.0, sample_weights[-1])
                        idx = bisect.bisect_left(sample_weights, rand_num)
                        context_and_negatives.append(idx)
                    yield right_idx, context_and_negatives

    def _gen_training_sentences(self) -> Iterator[str]:
        random_order = True
        if self._training_epoch < _NUM_EPOCHS_FOR_WORDS:
            # For phase 1.
            yield from corpus.gen_shixuehanying_words(random_order)
        else:
            # For phase 2.
            yield from corpus.gen_poem_sentences(random_order)


class Vocab:
    """A set of Chinese chars and their embeddings."""

    embedding_dim: int
    _counts: list[int]
    _chars: list[str]
    _indices: dict[str, int]
    _embeddings: nn.Embedding

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        if not os.path.exists(_VOCAB_PATH):
            self._gen_vocab()
        else:
            self._load_vocab()
        trainer = EmbeddingTrainer(self.embedding_dim, self._indices,
                                   self._counts)
        self._embeddings = trainer.gen_embeddings()
        self._embeddings.requires_grad_(False)
        self._embeddings.weight[-1] = -torch.mean(self._embeddings.weight[:-1],
                                                  dim=0)

    def __len__(self) -> int:
        return len(self._chars)

    def __getitem__(self, index: int) -> str:
        return self._chars[index]

    def __contains__(self, ch: str) -> bool:
        return ch in self._indices

    def get_index(self, ch: str) -> int | None:
        return self._indices.get(ch)

    def get_index_with_default(self, ch: str) -> int:
        vocab_size = len(self._chars)
        return self._indices.get(ch, vocab_size - 1)

    def get_index_as_tensor(self, text: str) -> torch.Tensor:
        vocab_size = len(self._chars)
        tensor = torch.zeros(len(text), dtype=torch.long)
        for i, ch in enumerate(text):
            tensor[i] = self._indices.get(ch, vocab_size - 1)
        return tensor

    def embedding(self, x: str | torch.Tensor) -> torch.Tensor:
        vocab_size = len(self._chars)
        if isinstance(x, str):
            return self._embeddings(self.get_index_as_tensor(x))
        return self._embeddings(x)

    def embedding_matrix(self) -> torch.Tensor:
        return self._embeddings.weight

    def get_count_at_index(self, idx) -> int:
        return self._counts[idx]

    def _gen_vocab(self) -> None:
        print('Generating vocabulary ...')
        count_dict: dict[str, int] = {}
        for sentence in corpus.gen_all_sentences():
            for ch in sentence:
                count_dict[ch] = count_dict.get(ch, 0) + 1
        count_ch_pairs = sorted(((count, ch)
                                 for ch, count in count_dict.items()
                                 if count >= _MIN_WORD_COUNT),
                                reverse=True)
        count_ch_pairs.append((0, END_OF_SENTENCE))
        self._chars = []
        self._counts = []
        self._indices = {}
        for count, ch in count_ch_pairs:
            self._indices[ch] = len(self._chars)
            self._chars.append(ch)
            self._counts.append(count)
        with open(_VOCAB_PATH, 'w') as fout:
            for char, count in zip(self._chars, self._counts):
                fout.write(f'{char}\t{count}\n')
        print('Finished generating vocabulary.')

    def _load_vocab(self) -> None:
        self._counts = []
        self._chars = []
        with open(_VOCAB_PATH, 'r') as fin:
            for line in fin.readlines():
                toks = line.strip().split('\t')
                if len(toks) != 2:
                    raise Exception(f'Expected 2 tokens but got {len(toks)}.')
                self._chars.append(toks[0])
                self._counts.append(int(toks[1]))
        self._indices = {}
        for idx, char in enumerate(self._chars):
            self._indices[char] = idx



if __name__ == '__main__':
    common.global_init()
    vocab = Vocab(EMBEDDING_DIM)
    # A quick demo on the embedding quality.
    for char in '风花雪月鸟鱼虫':
        idx = vocab.get_index(char)
        if idx is None:
            print('Not found')
            continue
        candidates = []
        wordvec = vocab.embedding(char).squeeze(0)
        for other_idx in range(len(vocab) - 1):
            if idx == other_idx:
                continue
            othervec = vocab.embedding(vocab[other_idx]).squeeze(0)
            score = torch.sum(wordvec * othervec)
            heapq.heappush(candidates, (score, vocab[other_idx]))
            if len(candidates) > 5:
                heapq.heappop(candidates)
        candidates.sort(reverse=True)
        candidate_str = ''.join(ch for _, ch in candidates)
        print(f'{char}: {candidate_str}')
