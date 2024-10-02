import torch
from typing import Protocol


class FeatureExtractorDelegate(Protocol):
    features: torch.nn.Module
