import torch
from typing import NamedTuple


class FeaturesData(NamedTuple):
    """The wrapped data structured for features"""
    z: torch.Tensor
    """The un-quantized features"""
    z_q: torch.Tensor
    """The quantized features"""
