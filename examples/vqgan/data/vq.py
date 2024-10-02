import torch
from typing import NamedTuple

from .discrimination import DiscrminatedData
from .features import FeaturesData


class VQData(NamedTuple):
    """The wrapped data structure for VQGAN"""
    y: torch.Tensor
    """The final output of VQGAN"""
    f: FeaturesData
    """The non-quantized and quantized features in `.features.FeaturesData`"""
    d: DiscrminatedData
    """The discriminated logits in `.discrimination.DiscriminatedData`"""
