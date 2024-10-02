import torch
from typing import NamedTuple


class DiscrminatedData(NamedTuple):
    """The wrapped data structure for discriminated logits"""
    fake: torch.Tensor
    """The fake discriminated logits"""
    real: torch.Tensor
    """The real discriminated logits"""
