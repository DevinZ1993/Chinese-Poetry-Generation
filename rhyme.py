#! /usr/bin/env python3
# -*- coding:utf-8 -*-
"""A module that manages the rhyming feature."""

import common
import corpus
import vocab

INFINITE_COST: int = 1_000_000_000
RHYMING_COST_WEIGHT: int = 14

_DISPLAY_DEBUG_INFO: bool = False


def _split_pinyin(pinyin: str) -> tuple[str, str]:
    consonants = 'BPMFDTNLGKHJQXZCSRYW'
    if pinyin.startswith('ZH') or pinyin.startswith('CH') or pinyin.startswith(
            'SH'):
        consonant = pinyin[:2]
        vowel = pinyin[2:]
    elif pinyin[0] in consonants:
        consonant = pinyin[0]
        vowel = pinyin[1:]
    else:
        consonant = ''
        vowel = pinyin
    if consonant in 'JQXY' and vowel[0] == 'U':
        vowel = 'V' + vowel[1:]
    if consonant in ['ZH', 'CH', 'SH', 'R'] and vowel == 'I':
        vowel = 'HI'
    if consonant == 'Y' and vowel == 'E':
        vowel = 'IE'
    return consonant, vowel


_VOWEL_GROUPS: list[list[str]] = [['EN', 'IN', 'UN', 'VN'],
                                  ['ENG', 'ING', 'ONG', 'IONG', 'UENG'],
                                  ['A', 'IA', 'UA'], ['E', 'O', 'UO'],
                                  ['I', 'V'], ['AI', 'UAI'],
                                  ['ANG', 'IANG', 'UANG'], ['EI', 'UI', 'UEI'],
                                  ['IE', 'VE'], ['AN', 'IAN', 'UAN', 'VAN'],
                                  ['AO', 'IAO'], ['IU', 'OU', 'IOU']]

_WUYANJUEJU_PATTERNS: list[list[str]] = [
    ['NZPPZ', 'PPZZP', 'NPPZZ', 'NZZPP'],
    ['NZZPP', 'PPZZP', 'NPPZZ', 'NZZPP'],
    ['NPPZZ', 'NZZPP', 'NZPPZ', 'PPZZP'],
    ['PPZZP', 'NZZPP', 'NZPPZ', 'PPZZP'],
]

_QIYANJUEJU_PATTERNS: list[list[str]] = [
    ['NPNZZPP', 'NZPPZZP', 'NZNPPZZ', 'NPNZZPP'],
    ['NPNZPPZ', 'NZPPZZP', 'NZNPPZZ', 'NPNZZPP'],
    ['NZPPZZP', 'NPNZZPP', 'NPNZPPZ', 'NZPPZZP'],
    ['NZNPPZZ', 'NPNZZPP', 'NPNZPPZ', 'NZPPZZP'],
]

_WUYANLVSHI_PATTERNS: list[list[str]] = [
    ['NZPPZ', 'PPZZP', 'NPPZZ', 'NZZPP', 'NZPPZ', 'PPZZP', 'NPPZZ', 'NZZPP'],
    ['NZZPP', 'PPZZP', 'NPPZZ', 'NZZPP', 'NZPPZ', 'PPZZP', 'NPPZZ', 'NZZPP'],
    ['NPPZZ', 'NZZPP', 'NZPPZ', 'PPZZP', 'NPPZZ', 'NZZPP', 'NZPPZ', 'PPZZP'],
    ['PPZZP', 'NZZPP', 'NZPPZ', 'PPZZP', 'NPPZZ', 'NZZPP', 'NZPPZ', 'PPZZP'],
]

_QIYANLVSHI_PATTERNS: list[list[str]] = [
    [
        'NPNZZPP', 'NZPPZZP', 'NZNPPZZ', 'NPNZZPP', 'NPNZPPZ', 'NZPPZZP',
        'NZNPPZZ', 'NPNZZPP'
    ],
    [
        'NPNZPPZ', 'NZPPZZP', 'NZNPPZZ', 'NPNZZPP', 'NPNZPPZ', 'NZPPZZP',
        'NZNPPZZ', 'NPNZZPP'
    ],
    [
        'NZPPZZP', 'NPNZZPP', 'NPNZPPZ', 'NZPPZZP', 'NZNPPZZ', 'NPNZZPP',
        'NPNZPPZ', 'NZPPZZP'
    ],
    [
        'NZNPPZZ', 'NPNZZPP', 'NPNZPPZ', 'NZPPZZP', 'NZNPPZZ', 'NPNZZPP',
        'NPNZPPZ', 'NZPPZZP'
    ],
]


class Rhyme:
    """A class that offers rule-based evaluation on the rhyming of a poem."""

    def __init__(self):
        self.ping_chars: set[int] = set()
        self.ze_chars: set[int] = set()
        self.vocab = vocab.Vocab(vocab.EMBEDDING_DIM)
        self.rhyme_groups: list[set[str]] = [set() for _ in self.vocab]
        vowel_roots: dict[str, str] = {}
        for group in _VOWEL_GROUPS:
            for vowel in group[1:]:
                vowel_roots[vowel] = group[0]
        with open(corpus.PINYIN_PATH, 'r') as fin:
            for line in fin.readlines():
                toks = [tok for tok in line.strip().split(' ') if tok]
                assert len(toks) > 1
                ch = chr(int(toks[0], 16))
                ch_idx = self.vocab.get_index(ch)
                if ch_idx is None:
                    continue
                for tok in toks[1:]:
                    if tok[-1] < '1' or tok[-1] > '4':
                        continue
                    if tok[-1] <= '2':
                        self.ping_chars.add(ch_idx)
                    else:
                        self.ze_chars.add(ch_idx)
                    _, vowel = _split_pinyin(tok[:-1])
                    vowel = vowel_roots.get(vowel, vowel)
                    self.rhyme_groups[ch_idx].add(vowel)
        with open(corpus.PSY_PATH, 'r') as fin:
            is_head = True
            for line in fin.readlines():
                if is_head:
                    head = line.strip()
                    is_head = False
                    continue
                hidden = False
                for ch in line.strip():
                    if hidden:
                        if ch == ']' or ch == '>':
                            hidden = False
                        continue
                    if ch == '[' or ch == '<':
                        hidden = True
                        continue
                    ch_idx = self.vocab.get_index(ch)
                    if ch_idx is None:
                        continue
                    if head[1] == '平':
                        self.ping_chars.add(ch_idx)
                    else:
                        self.ze_chars.add(ch_idx)
                    self.rhyme_groups[ch_idx].add(head)
                assert not hidden
                is_head = True

    def eval(self, poem: list[str]) -> int:
        min_cost = (1 << 31) - 1
        for rule in _WUYANJUEJU_PATTERNS:
            min_cost = min(min_cost, self.eval_on_single_rule(poem, rule))
        for rule in _QIYANJUEJU_PATTERNS:
            min_cost = min(min_cost, self.eval_on_single_rule(poem, rule))
        for rule in _WUYANLVSHI_PATTERNS:
            min_cost = min(min_cost, self.eval_on_single_rule(poem, rule))
        for rule in _QIYANLVSHI_PATTERNS:
            min_cost = min(min_cost, self.eval_on_single_rule(poem, rule))
        return min_cost

    def eval_on_single_rule(self, poem: list[str], rule: list[str]) -> int:
        max_cost: int = INFINITE_COST
        if len(poem) != len(rule) or any(
                len(sentence) != len(rule_row)
                for sentence, rule_row in zip(poem, rule)):
            return max_cost
        cost: int = 0
        rhymes: list[int] = []
        for sentence, rule_row in zip(poem, rule):
            for ch, pz in zip(sentence, rule_row):
                ch_idx = self.vocab.get_index(ch)
                if ch_idx is None:
                    continue
                if ((pz == 'P' and ch_idx not in self.ping_chars) or
                    (pz == 'Z' and ch_idx not in self.ze_chars)):
                    cost += 1
                    if _DISPLAY_DEBUG_INFO:
                        print(f'Char {ch} violated {pz}.')
            if rule_row[-1] == 'P':
                ch_idx = self.vocab.get_index(sentence[-1])
                assert ch_idx is not None
                rhymes.append(ch_idx)
        group_candidates: dict[str, int] = {}
        for ch_idx in rhymes:
            for group in self.rhyme_groups[ch_idx]:
                group_candidates[group] = group_candidates.get(group, 0) + 1
        max_count = max(cnt for _, cnt in group_candidates.items())
        rhyme_cost: int = len(rhymes) - max_count
        if rhyme_cost > 0:
            cost += rhyme_cost * RHYMING_COST_WEIGHT
            if _DISPLAY_DEBUG_INFO:
                print(f'Rhyming cost: {rhyme_cost}.')
        return cost


def _test_rhyming_pattern(rhyme: Rhyme,
                          poem: str,
                          rule: list[str],
                          expected_cost: int = 0) -> None:
    sentences = []
    i = 0
    for j in range(len(poem) + 1):
        if j < len(poem) and common.is_cn_char(poem[j]):
            continue
        if i < j:
            sentences.append(poem[i:j])
        i = j + 1
    assert rhyme.eval_on_single_rule(sentences, rule) == expected_cost


if __name__ == '__main__':
    for rule in _WUYANJUEJU_PATTERNS:
        assert len(rule) == 4
        assert all(len(row) == 5 for row in rule)
    for rule in _QIYANJUEJU_PATTERNS:
        assert len(rule) == 4
        assert all(len(row) == 7 for row in rule)
    for rule in _WUYANLVSHI_PATTERNS:
        assert len(rule) == 8
        assert all(len(row) == 5 for row in rule)
    for rule in _QIYANLVSHI_PATTERNS:
        assert len(rule) == 8
        assert all(len(row) == 7 for row in rule)
    rhyme = Rhyme()
    _test_rhyming_pattern(rhyme, '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。',
                          _WUYANJUEJU_PATTERNS[0])
    _test_rhyming_pattern(rhyme, '寥落古行宫，宫花寂寞红。白头宫女在，闲坐说玄宗。',
                          _WUYANJUEJU_PATTERNS[1])
    _test_rhyming_pattern(rhyme, '鸣筝金粟柱，素手玉房前。欲得周郎顾，时时误拂弦。',
                          _WUYANJUEJU_PATTERNS[2])
    _test_rhyming_pattern(rhyme, '花明绮陌春，柳拂御沟新。为报辽阳客，流芳不待人。',
                          _WUYANJUEJU_PATTERNS[3])
    _test_rhyming_pattern(rhyme, '朝辞白帝彩云间，千里江陵一日还。 两岸猿声啼不住，轻舟已过万重山。',
                          _QIYANJUEJU_PATTERNS[0])
    _test_rhyming_pattern(rhyme, '曾栽杨柳江南岸， 一别江南两度春。 遥忆青青江岸上，不知攀折是何人。',
                          _QIYANJUEJU_PATTERNS[1])
    _test_rhyming_pattern(rhyme, '君问归期未有期，巴山夜雨涨秋池。何当共剪西窗烛，却话巴山夜雨时。',
                          _QIYANJUEJU_PATTERNS[2])
    _test_rhyming_pattern(rhyme, '独在异乡为异客， 每逢佳节倍思亲。遥知兄弟登高处， 遍插茱萸少一人。',
                          _QIYANJUEJU_PATTERNS[3])
    _test_rhyming_pattern(
        rhyme, """
               国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。
               烽火连三月，家书抵万金。白头掻更短，浑欲不胜簪。
               """, _WUYANLVSHI_PATTERNS[0])
    _test_rhyming_pattern(
        rhyme, """
               红叶晚萧萧，长亭酒一瓢。残云归太华，疏雨过中条。
               树色随山迥，河声入海遥。帝乡明日到，犹自梦渔樵。
               """, _WUYANLVSHI_PATTERNS[1])
    _test_rhyming_pattern(
        rhyme, """
               空山新雨后，天气晚来秋。明月松间照，清泉石上流。
               竹喧归浣女，莲动下渔舟。随意春芳歇，王孙自可留。
               """, _WUYANLVSHI_PATTERNS[2])
    _test_rhyming_pattern(
        rhyme, """
               深居俯夹城，春去夏犹清。天意怜幽草，人间重晚晴。
               并添高阁迥，微注小窗明。越鸟巢乾后，归飞体更轻。
               """, _WUYANLVSHI_PATTERNS[3])
    _test_rhyming_pattern(
        rhyme, """
               一封朝奏九重天，夕贬潮州路八千。
               欲为圣明除弊事，肯将衰朽惜残年。
               云横秦岭家何在？雪拥蓝关马不前。
               知汝远来应有意，好收吾骨瘴江边。
               """, _QIYANLVSHI_PATTERNS[0])
    _test_rhyming_pattern(
        rhyme, """
               巴山楚水凄凉地，二十三年弃置身。
               怀旧空吟闻笛赋，到乡翻似烂柯人。
               沉舟侧畔千帆过，病树前头万木春。
               今日听君歌一曲，暂凭杯酒长精神。
               """, _QIYANLVSHI_PATTERNS[1])
    _test_rhyming_pattern(
        rhyme, """
               锦瑟无端五十弦，一弦一柱思华年。
               庄生晓梦迷蝴蝶，望帝春心托杜鹃。
               沧海月明珠有泪，蓝田日暖玉生烟。
               此情可待成追忆，只是当时已惘然。
               """, _QIYANLVSHI_PATTERNS[2])
    _test_rhyming_pattern(
        rhyme, """
               岁暮阴阳催短景，天涯霜雪霁寒宵。
               五更鼓角声悲壮，三峡星河影动摇。
               野哭千家闻战伐，夷歌数处起渔樵。
               卧龙跃马终黄土，人事音书漫寂寥。
               """, _QIYANLVSHI_PATTERNS[3])
