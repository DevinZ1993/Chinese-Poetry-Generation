#! /usr/bin/env python3
# -*- coding:utf-8 -*-
"""A neural poem generator"""

import numpy as np
import matplotlib.pyplot as plt

import dataclasses
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
from torch.utils import tensorboard
from typing import Any, Iterator

import common
import corpus
import rhyme
import vocab

EMBEDDING_DIM: int = vocab.EMBEDDING_DIM
MODEL_DIM: int = EMBEDDING_DIM

_MODEL_PATH: str = os.path.join(corpus.GENDATA_PATH, 'lstm_500.pt')
_MAX_CONTEXT_SENTENCES: int = 3
_NUM_RNN_LAYERS: int = 4
_ALL_POEM_EPOCHS: int = 2
_QIYANSHI_EPOCHS: int = 4
_TOTAL_NUM_EPOCHS: int = _ALL_POEM_EPOCHS + _QIYANSHI_EPOCHS
_BATCH_SIZE: int = 64
_NUM_BATCHES_FOR_LOGGING: int = 10
_ALL_POEM_EOS_WEIGHT: float = 1.0
_QIYANSHI_EOS_WEIGHT: float = 10.0
_ENCODER_LEARNING_RATE: float = 0.002
_DECODER_LEARNING_RATE: float = 0.002
_FORCE_TRAINING: bool = False
_MAX_SENTENCE_LEN: int = 11
_BEAM_SIZE: int = 3
_VISUALIZE_ATTENTION_WEIGHTS: bool = False
_TEACHER_FORCING: bool = False
_ENABLE_EOS_HACK: bool = False


class Encoder(nn.Module):

    def __init__(self, vocab_instance: vocab.Vocab):
        super(Encoder, self).__init__()
        self.vocab = vocab_instance
        self.dropout = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                            hidden_size=MODEL_DIM,
                            proj_size=0,
                            num_layers=_NUM_RNN_LAYERS,
                            batch_first=True,
                            bidirectional=True)
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=_ENCODER_LEARNING_RATE)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_length = inputs.shape  # B x M

        if seq_length:
            embeddings = self.dropout(self.vocab.embedding(inputs))
            outputs, (hidden_state, cell_state) = self.lstm(embeddings)
        else:
            outputs = torch.zeros(batch_size, 0, 2 * EMBEDDING_DIM)
            hidden_state = torch.zeros(2 * _NUM_RNN_LAYERS, batch_size,
                                       EMBEDDING_DIM)
            cell_state = torch.zeros(2 * _NUM_RNN_LAYERS, batch_size, MODEL_DIM)

        assert outputs.shape == (batch_size, seq_length, 2 * EMBEDDING_DIM)
        assert hidden_state.shape == (2 * _NUM_RNN_LAYERS, batch_size,
                                      EMBEDDING_DIM)
        assert cell_state.shape == (2 * _NUM_RNN_LAYERS, batch_size, MODEL_DIM)

        return outputs, (hidden_state[:_NUM_RNN_LAYERS, :, :],
                         cell_state[:_NUM_RNN_LAYERS, :, :])


def _visualize_attention_weights(attention_weights: torch.Tensor) -> None:
    attention_weights = np.array(attention_weights)
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Attention Weight")
    plt.ylabel("Batch")
    plt.xlabel("Encoder Output Token")
    plt.title("Attention Heatmap")
    plt.show()


class BahdanauAttention(nn.Module):

    def __init__(self):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(EMBEDDING_DIM, MODEL_DIM)
        self.Ua = nn.Linear(2 * EMBEDDING_DIM, MODEL_DIM)
        self.Va = nn.Linear(MODEL_DIM, 1)
        nn.init.xavier_uniform_(self.Wa.weight)
        nn.init.xavier_uniform_(self.Ua.weight)
        nn.init.xavier_uniform_(self.Va.weight)
        nn.init.zeros_(self.Wa.bias)
        nn.init.zeros_(self.Ua.bias)
        nn.init.zeros_(self.Va.bias)

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                key_padding_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, key_seq_length = key_padding_mask.shape  # B x M
        # The keys are the bidirectional RNN encoder outputs.
        assert keys.shape == (batch_size, key_seq_length, 2 * EMBEDDING_DIM)
        # The query is the top-layer decoder hidden state.
        assert query.shape == (batch_size, EMBEDDING_DIM)  # B x d
        scores = self.Va(
            torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys))).permute(
                0, 2, 1)  # B x 1 x M
        scores.masked_fill_(key_padding_mask.unsqueeze(1), float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        if _VISUALIZE_ATTENTION_WEIGHTS:
            _visualize_attention_weights(attention_weights.detach().squeeze(1))
        context = torch.bmm(attention_weights, keys)
        assert context.shape == (batch_size, 1, 2 * EMBEDDING_DIM)
        return context


class Decoder(nn.Module):

    def __init__(self, vocab_instance: vocab.Vocab):
        super(Decoder, self).__init__()
        self.vocab = vocab_instance
        self.output_projection = self.vocab.embedding_matrix().permute(
            1, 0)  # D x V
        self.lstm = nn.LSTM(input_size=3 * EMBEDDING_DIM,
                            hidden_size=MODEL_DIM,
                            proj_size=0,
                            num_layers=_NUM_RNN_LAYERS,
                            batch_first=True)
        self.attention = BahdanauAttention()
        self.dropout = nn.Dropout(p=0.1)
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=_DECODER_LEARNING_RATE)

    def forward(
        self, encoder_outputs: torch.Tensor,
        encoder_output_padding_mask: torch.BoolTensor, inputs: torch.Tensor,
        hidden_and_cell_states: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, input_seq_length = encoder_output_padding_mask.shape
        assert encoder_outputs.shape == (batch_size, input_seq_length,
                                         2 * EMBEDDING_DIM)
        assert hidden_and_cell_states[0].shape == (_NUM_RNN_LAYERS, batch_size,
                                                   EMBEDDING_DIM)  # L x B x d
        assert hidden_and_cell_states[1].shape == (_NUM_RNN_LAYERS, batch_size,
                                                   MODEL_DIM)  # L x B x D
        assert inputs.shape == (batch_size,)

        embeddings = self.vocab.embedding(inputs).unsqueeze(1)  # B x 1 x d
        context = self.attention(
            query=hidden_and_cell_states[0][-1],
            keys=encoder_outputs,
            key_padding_mask=encoder_output_padding_mask)  # B x 1 x 2d
        lstm_inputs = self.dropout(torch.cat((embeddings, context),
                                             dim=-1))  # B x 1 x 3d
        lstm_outputs, hidden_and_cell_states = self.lstm(
            lstm_inputs, hidden_and_cell_states)
        projected_outputs = torch.matmul(lstm_outputs,
                                         self.output_projection)  # B x 1 x V
        outputs = F.log_softmax(projected_outputs.squeeze(1), dim=-1)
        return outputs, hidden_and_cell_states


def load_model(model: Encoder | Decoder, prefix: str,
               checkpoint: dict[str, Any]) -> None:
    print(f'Loading model for {prefix} ...')
    model.load_state_dict(checkpoint[f'{prefix}_model'])
    model.optimizer.load_state_dict(checkpoint[f'{prefix}_optimizer'])


def save_model(model: Encoder | Decoder, prefix: str,
               checkpoint: dict[str, Any]) -> None:
    checkpoint[f'{prefix}_model'] = model.state_dict()
    checkpoint[f'{prefix}_optimizer'] = model.optimizer.state_dict()


def grad_norm(model: Encoder | Decoder) -> float:
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item()**2
    return math.sqrt(total_norm)


@dataclasses.dataclass
class BeamSearchState:
    prefix: str
    log_likelihood: float
    hidden_and_cell_states: tuple[torch.Tensor, torch.Tensor]

    def __lt__(self, other) -> bool:
        return self.log_likelihood < other.log_likelihood


class Generator:
    """"An LSTM encoder-decoder poem generator."""

    def __init__(self):
        self.vocab = vocab.Vocab(EMBEDDING_DIM)
        self.vocab_size = len(self.vocab)
        self.encoder = Encoder(self.vocab)
        self.decoder = Decoder(self.vocab)
        self._training_epoch = 0
        self._training_step = 0
        if not _FORCE_TRAINING and os.path.exists(_MODEL_PATH):
            self._load_model()
        self.summary_writer = tensorboard.SummaryWriter(
            'gen_data/lstm_gradient')
        self.global_batch_idx = 0
        if self._training_epoch < _TOTAL_NUM_EPOCHS:
            self._train_model()
        self.encoder.eval()
        self.decoder.eval()

    def _load_model(self) -> None:
        checkpoint = torch.load(_MODEL_PATH, weights_only=True)
        self._training_epoch = checkpoint.get('epoch', 0)
        self._training_step = checkpoint.get('step', 0)
        load_model(self.encoder, 'encoder', checkpoint)
        load_model(self.decoder, 'decoder', checkpoint)

    def _save_model(self) -> None:
        checkpoint_path = f'{_MODEL_PATH}.tmp'
        checkpoint = {
            'epoch': self._training_epoch,
            'step': self._training_step,
        }
        save_model(self.encoder, 'encoder', checkpoint)
        save_model(self.decoder, 'decoder', checkpoint)
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint {checkpoint_path}.')
        os.replace(checkpoint_path, _MODEL_PATH)

    def _train_model(self) -> None:
        """Trains the model in the teacher forcing fashion.

        The training happens in two phases:
        1. In the first phase, we use all the poems across the entire corpus.
        2. In the second phase, we focus on poems with 7 chars per line (七言诗).
        """

        print('Training the LSTM poem generator ...')
        self.encoder.train()
        self.decoder.train()
        self.target_weight = torch.ones(self.vocab_size)
        while self._training_epoch < _TOTAL_NUM_EPOCHS:
            print(f'Epoch #{self._training_epoch} ...')
            loss_vals: list[float] = []
            encoder_grad_norm_sum: float = 0.0
            decoder_grad_norm_sum: float = 0.0
            for data_batch in self._get_training_data_batches():
                self.encoder.optimizer.zero_grad()
                self.decoder.optimizer.zero_grad()
                loss_vals.append(self._calculate_batch_mean_loss(data_batch))
                self.encoder.optimizer.step()
                self.decoder.optimizer.step()
                encoder_grad_norm_sum += grad_norm(self.encoder)
                decoder_grad_norm_sum += grad_norm(self.decoder)
                if len(loss_vals) % _NUM_BATCHES_FOR_LOGGING == 0:
                    loss_val = statistics.mean(
                        loss_vals[-_NUM_BATCHES_FOR_LOGGING:])
                    print('Epoch #{}, {} batches: loss = {}'.format(
                        self._training_epoch, len(loss_vals), loss_val))
                    encoder_grad_norm = encoder_grad_norm_sum / _NUM_BATCHES_FOR_LOGGING
                    encoder_grad_norm_sum = 0.0
                    print(f'\tencoder grad: {encoder_grad_norm}')
                    decoder_grad_norm = decoder_grad_norm_sum / _NUM_BATCHES_FOR_LOGGING
                    decoder_grad_norm_sum = 0.0
                    print(f'\tdecoder grad: {decoder_grad_norm}')
                self._training_step += 1
            mean_loss = statistics.mean(loss_vals)
            print(
                'Finished epoch #{} ({} batches in total): loss = {} (stdev={})'
                .format(self._training_epoch, len(loss_vals), mean_loss,
                        statistics.stdev(loss_vals)))
            self._training_epoch += 1
            self._save_model()
        print('Finished training the LSTM poem generator.')

    def _calculate_batch_mean_loss(self, data_batch: list[tuple[str,
                                                                str]]) -> float:
        # Prepare model inputs.
        batch_size = len(data_batch)
        src_length = 0
        target_length = 0
        for context, sentence in data_batch:
            src_length = max(src_length, len(context))
            target_length = max(target_length, len(sentence))

        eos_idx = self.vocab.get_index_with_default(vocab.END_OF_SENTENCE)
        encoder_inputs = torch.full((batch_size, src_length),
                                    eos_idx,
                                    dtype=torch.long)
        encoder_output_padding_mask = torch.zeros(batch_size,
                                                  src_length,
                                                  dtype=torch.bool)
        decoder_inputs = torch.full((batch_size, target_length),
                                    eos_idx,
                                    dtype=torch.long)
        target = -torch.ones(batch_size, target_length, dtype=torch.long)
        for poem_idx, (context, sentence) in enumerate(data_batch):
            encoder_inputs[
                poem_idx, :len(context)] = self.vocab.get_index_as_tensor(
                    context)
            encoder_output_padding_mask[poem_idx, len(context):] = True
            decoder_inputs[poem_idx, :len(sentence) -
                           1] = self.vocab.get_index_as_tensor(sentence[:-1])
            target[poem_idx, :len(sentence) -
                   1] = self.vocab.get_index_as_tensor(sentence[1:])

        # Run the language model.
        encoder_outputs, hidden_and_cell_states = self.encoder(encoder_inputs)

        logit_list = []
        for idx in range(target_length):
            logits, hidden_and_cell_states = self.decoder(
                encoder_outputs=encoder_outputs,
                encoder_output_padding_mask=encoder_output_padding_mask,
                inputs=decoder_inputs[:, idx],
                hidden_and_cell_states=hidden_and_cell_states)
            logit_list.append(logits.unsqueeze(1))
        logits = torch.cat(logit_list, dim=1)
        assert logits.shape == (batch_size, target_length, self.vocab_size)

        # Calculate the loss.
        loss_weight = self.target_weight
        loss_weight[self.vocab.get_index_with_default(
            vocab.END_OF_SENTENCE)] = (_ALL_POEM_EOS_WEIGHT
                                       if self._training_epoch else
                                       _QIYANSHI_EOS_WEIGHT)
        loss_function = nn.NLLLoss(weight=loss_weight, ignore_index=-1)
        loss = loss_function(logits.view(-1, self.vocab_size), target.view(-1))
        loss.backward()
        self._track_model_gradients()
        nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=5.0)
        return loss.item()

    def _get_training_data_batches(self) -> Iterator[list[tuple[str, str]]]:
        buffer: list[tuple[str, str]] = []
        for context_sentence_pair in self._get_training_data():
            buffer.append(context_sentence_pair)
            if len(buffer) == _BATCH_SIZE:
                yield buffer
                buffer = []
        if buffer:
            yield buffer

    def _get_training_data(self) -> Iterator[tuple[str, str]]:
        if self._training_epoch < _ALL_POEM_EPOCHS:
            poem_filter = lambda p: True
        else:
            poem_filter = lambda p: corpus.is_qiyanjueju(
                p) or corpus.is_qiyanlvshi(p)
        poems = list(
            filter(
                poem_filter,
                corpus.get_poems(lambda ch: ch in self.vocab,
                                 random_order=True)))
        total_batches = math.ceil(
            sum(len(poem) for poem in poems) / _BATCH_SIZE)
        print(f'Batches per epoch: {total_batches}')

        for poem in poems:
            for i, sentence in enumerate(poem):
                context_start = max(0, i - _MAX_CONTEXT_SENTENCES)
                sentence += vocab.END_OF_SENTENCE
                if i == 0:
                    yield vocab.END_OF_SENTENCE, sentence
                else:
                    yield ''.join(poem[context_start:i]), sentence

    def _track_model_gradients(self) -> None:
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                self.summary_writer.add_histogram(f'encoder/{name}.grad',
                                                  param.grad,
                                                  self.global_batch_idx)
        for name, param in self.decoder.named_parameters():
            if param.grad is not None:
                self.summary_writer.add_histogram(f'decoder/{name}.grad',
                                                  param.grad,
                                                  self.global_batch_idx)
        self.global_batch_idx += 1

    def generate(self, sentence_heads: str) -> list[str]:
        """Generates a poem given the first character of each sentence (藏头诗)."""

        sentences: list[str] = []
        for idx, sentence_head in enumerate(sentence_heads):
            context = ''.join(
                sentence + vocab.END_OF_SENTENCE for sentence in
                sentences[max(0,
                              len(sentences) - _MAX_CONTEXT_SENTENCES):])
            if not context:
                context = vocab.END_OF_SENTENCE
            with torch.no_grad():
                # Run the encoder on the preceding sentences.
                encoder_outputs, hidden_and_cell_states = self.encoder(
                    self.vocab.get_index_as_tensor(context).unsqueeze(0))
                encoder_output_padding_mask = torch.zeros(1,
                                                          len(context),
                                                          dtype=torch.bool)
                # Do beam search to generate the next sentence starting with
                # `sentence_head`.
                state_candidates: list[BeamSearchState] = [
                    BeamSearchState(sentence_head, 0.0, hidden_and_cell_states)
                ]
                for pos in range(len(sentence_head), _MAX_SENTENCE_LEN):
                    next_candidates: list[BeamSearchState] = []
                    for parent_state in state_candidates:
                        if parent_state.prefix[-1] == vocab.END_OF_SENTENCE:
                            heapq.heappush(next_candidates, parent_state)
                            if len(next_candidates) > _BEAM_SIZE:
                                heapq.heappop(next_candidates)
                            continue
                        # Run the decoder on the last generated char.
                        inputs = self.vocab.get_index_as_tensor(
                            parent_state.prefix[-1])
                        outputs, hidden_and_cell_states = self.decoder(
                            encoder_outputs=encoder_outputs,
                            encoder_output_padding_mask=
                            encoder_output_padding_mask,
                            inputs=inputs,
                            hidden_and_cell_states=parent_state.
                            hidden_and_cell_states)
                        log_probs = outputs[0, :]
                        if _ENABLE_EOS_HACK:
                            if pos == 7:
                                log_probs[:-1] = float('-inf')
                            else:
                                log_probs[-1] = float('-inf')
                        for ch_idx in range(self.vocab_size):
                            next_candidate = BeamSearchState(
                                parent_state.prefix + self.vocab[ch_idx],
                                parent_state.log_likelihood +
                                log_probs[ch_idx].item(),
                                hidden_and_cell_states)
                            heapq.heappush(next_candidates, next_candidate)
                            if len(next_candidates) > _BEAM_SIZE:
                                heapq.heappop(next_candidates)
                    state_candidates = next_candidates
                    if all(state.prefix[-1] == vocab.END_OF_SENTENCE
                           for state in state_candidates):
                        break
            state_candidates.sort()
            sentences.append(state_candidates[-1].prefix.replace(
                vocab.END_OF_SENTENCE, ''))
        return sentences

    def show_teacher_forcing(self, poem: list[str], k: int) -> None:
        """Displays the generated tokens in the teacher forcing mode."""

        sentences: list[str] = []
        for sentence in poem:
            context = ''.join(sentences) if sentences else vocab.END_OF_SENTENCE
            with torch.no_grad():
                encoder_outputs, hidden_and_cell_states = self.encoder(
                    self.vocab.get_index_as_tensor(context).unsqueeze(0))
                encoder_output_padding_mask = torch.zeros(1,
                                                          len(context),
                                                          dtype=torch.bool)
                for i in range(1, len(sentence) + 1):
                    inputs = self.vocab.get_index_as_tensor(sentence[i - 1])
                    outputs, hidden_and_cell_states = self.decoder(
                        encoder_outputs=encoder_outputs,
                        encoder_output_padding_mask=encoder_output_padding_mask,
                        inputs=inputs,
                        hidden_and_cell_states=hidden_and_cell_states)
                    log_probs = outputs[0, :]
                    candidates: list[tuple[float, str]] = []
                    for j, ch in enumerate(self.vocab):
                        heapq.heappush(candidates, (log_probs[j].item(), ch))
                        if len(candidates) > k:
                            heapq.heappop(candidates)
                    results: list[tuple[str, float]] = []
                    while candidates:
                        log_prob, ch = heapq.heappop(candidates)
                        results.append((ch, math.exp(log_prob)))
                    target_ch = sentence[i] if i < len(
                        sentence) else vocab.END_OF_SENTENCE
                    target_idx = self.vocab.get_index_with_default(target_ch)
                    target_prob = math.exp(log_probs[target_idx].item())
                    print('{}({}): {} {}'.format(sentence[:i], target_ch,
                                                 target_prob, results[::-1]))
            sentences.append(sentence + vocab.END_OF_SENTENCE)


if __name__ == '__main__':
    common.global_init()
    generator = Generator()
    rhyme = rhyme.Rhyme()
    for poem in corpus.get_poems(lambda ch: ch in generator.vocab,
                                 random_order=False):
        if not corpus.is_qiyanjueju(poem) and not corpus.is_qiyanlvshi(poem):
            continue
        if _TEACHER_FORCING:
            generator.show_teacher_forcing(poem, 5)
            input('')
            continue
        heads = ''.join(sentence[0] for sentence in poem)
        sentences: list[str] = generator.generate(heads)
        for sentence in sentences:
            print(sentence)
        rhyme_cost: int = rhyme.eval(sentences)
        print(f'(rhyme cost: {rhyme_cost})')
        input()  # Wait on ENTER
