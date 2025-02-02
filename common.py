#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import random
import torch

_SEED: int = 10086


def global_init():
    """Initializes the training pipeline."""

    random.seed(_SEED)
    torch.manual_seed(_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)


def is_cn_char(ch: str) -> bool:
    """Returns true iff a char is a Chinese char."""

    return u'\u4e00' <= ch <= u'\u9fa5'
