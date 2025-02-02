#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import random
from typing import Callable, Iterator

import common

ROOT_DIR: str = os.path.dirname(__file__)
CORPUS_PATH: str = os.path.join(ROOT_DIR, 'raw_data')
GENDATA_PATH: str = os.path.join(ROOT_DIR, 'gen_data')

SXHY_PATH: str = os.path.join(CORPUS_PATH, 'shixuehanying.txt')
PINYIN_PATH: str = os.path.join(CORPUS_PATH, 'pinyin.txt')
PSY_PATH: str = os.path.join(CORPUS_PATH, 'pingshuiyun.txt')
POEM_PATHS: tuple[str] = tuple(
    os.path.join(CORPUS_PATH, filename)
    for filename in ('qts_tab.txt', 'qsc_tab.txt', 'qss_tab.txt',
                     'qtais_tab.txt', 'yuan.all', 'ming.all', 'qing.all'))

MAX_SENTENCE_LEN: int = 20
MIN_NUM_SENTENCES: int = 4
MAX_NUM_SENTENCES: int = 16


def _gen_single_sentences(content: str) -> Iterator[str]:
    start_idx = 0
    for end_idx in range(len(content) + 1):
        if end_idx == len(content) or not common.is_cn_char(content[end_idx]):
            if end_idx - start_idx > MAX_SENTENCE_LEN:
                # Discard the portion that contains overly long sentences since
                # they are more likely malformed.
                return
            if start_idx < end_idx:
                yield content[start_idx:end_idx]
            start_idx = end_idx + 1


def _is_content_well_formed(content: str) -> bool:
    red_flags = {'【', '】', '《', '》', '（', '）'}
    for ch in content:
        if ch in red_flags or '0' <= ch <= '9':
            return False
    return True


def gen_poem_sentences(random_order: bool = False) -> Iterator[str]:
    poem_paths = list(POEM_PATHS)
    if random_order:
        random.shuffle(poem_paths)
    for file in poem_paths:
        with open(file, 'r') as fin:
            metadata = fin.readline().strip().split('\t')
            body_index = metadata.index('body')
            lines = fin.readlines()
            if random_order:
                random.shuffle(lines)
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) <= body_index:
                    continue
                content = parts[body_index]
                if not _is_content_well_formed(content):
                    continue
                yield from _gen_single_sentences(content)


def gen_shixuehanying_words(random_order: bool = False) -> Iterator[str]:
    with open(SXHY_PATH, 'r') as fin:
        lines = fin.readlines()
        if random_order:
            random.shuffle(lines)
        for line in lines:
            if line.startswith('<begin>') or line.startswith('<end>'):
                continue
            yield from _gen_single_sentences(line.strip())


def gen_all_sentences() -> Iterator[str]:
    yield from gen_shixuehanying_words()
    yield from gen_poem_sentences()


def get_poems(is_char_in_vocab: Callable[[str], bool],
              random_order: bool = False) -> Iterator[list[str]]:
    buffer: list[list[str]] = []
    for file in POEM_PATHS:
        with open(file, 'r') as fin:
            metadata = fin.readline().strip().split('\t')
            body_index = metadata.index('body')
            for line in fin.readlines():
                content = line.strip().split('\t')[body_index]
                if not _is_content_well_formed(content):
                    continue
                poem = []
                for sentence in _gen_single_sentences(content):
                    if any(not is_char_in_vocab(ch) for ch in sentence):
                        poem.clear()
                        continue
                    poem.append(sentence)
                    if len(poem) == MAX_NUM_SENTENCES:
                        # Chunk a long poem into granular pieces.
                        if random_order:
                            buffer.append(poem)
                        else:
                            yield poem
                        poem = []
                if len(poem) >= MIN_NUM_SENTENCES:
                    if random_order:
                        buffer.append(poem)
                    else:
                        yield poem
    if random_order:
        random.shuffle(buffer)
        yield from buffer


def is_wuyanjueju(poem: list[str]) -> bool:
    return tuple(len(sentence) for sentence in poem) == (5, 5, 5, 5)


def is_qiyanjueju(poem: list[str]) -> bool:
    return tuple(len(sentence) for sentence in poem) == (7, 7, 7, 7)


def is_wuyanlvshi(poem: list[str]) -> bool:
    return tuple(len(sentence) for sentence in poem) == tuple(
        5 for _ in range(8))


def is_qiyanlvshi(poem: list[str]) -> bool:
    return tuple(len(sentence) for sentence in poem) == tuple(
        7 for _ in range(8))


if __name__ == '__main__':
    sxhy_words = list(gen_shixuehanying_words())
    print('Total ShiXueHanYing words: {}'.format(len(sxhy_words)))
    poem_sentences = list(gen_poem_sentences())
    print('Total poem sentences: {}'.format(len(poem_sentences)))
    wuyanjueju: int = 0
    qiyanjueju: int = 0
    wuyanlvshi: int = 0
    qiyanlvshi: int = 0
    for poem in get_poems(lambda ch: common.is_cn_char(ch), random_order=False):
        if is_wuyanjueju(poem):
            wuyanjueju += 1
            continue
        if is_qiyanjueju(poem):
            qiyanjueju += 1
            continue
        if is_wuyanlvshi(poem):
            wuyanlvshi += 1
            continue
        if is_qiyanlvshi(poem):
            qiyanlvshi += 1
    print(f'五言绝句：{wuyanjueju} 首')
    print(f'七言绝句：{qiyanjueju} 首')
    print(f'五言律诗：{wuyanlvshi} 首')
    print(f'七言律诗：{qiyanlvshi} 首')
