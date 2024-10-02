from torchvision.transforms import *  # type: ignore

import torch


class CityscapesMappingLabel:
    """Label Mapping from 35 classes to 19 classes form Cityscapes"""

    _mapping: dict[int, int]

    def __init__(self) -> None:
        self._mapping = { 
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 1,
            8: 2,
            9: 0,
            10: 0,
            11: 3,
            12: 4,
            13: 5,
            14: 0,
            15: 0,
            16: 0,
            17: 6,
            18: 0,
            19: 7,
            20: 8,
            21: 9,
            22: 10,
            23: 11,
            24: 12,
            25: 13,
            26: 14,
            27: 15,
            28: 16,
            29: 0,
            30: 0,
            31: 17,
            32: 18,
            33: 19,
            -1: 0
        }

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        label_mask = torch.zeros_like(label)
        for k in self._mapping:
            label_mask[label == k] = self._mapping[k]
        return label_mask
