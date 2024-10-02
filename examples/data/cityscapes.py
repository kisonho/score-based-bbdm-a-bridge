import torch
from torch._C import device
from torchmanager.data import Dataset
from torchmanager_core import devices
from torchmanager_core.typing import Optional, Union
from torchvision.datasets import Cityscapes as _Cityscapes
from torchvision.transforms import v2 as transforms

from .split import DatasetSplit as CityscapesSplit
from .transforms import CityscapesMappingLabel


class Cityscapes(_Cityscapes):
    """Dataset for the cityscapes dataset."""
    return_file_name: bool

    def __init__(self, root: str, /, img_size: Union[int, tuple[int, int]], split: CityscapesSplit = CityscapesSplit.TRAIN, *, repeat: int = 1, return_file_name: bool = False) -> None:
        """
        Constructor

        - Parameters:
            - root: a `str` containing the path to the root directory of the dataset.
            - img_size: an `int` or a `tuple` of `int` containing the size of the image.
            - split: a `CityscapesSplit` indicating which split to use.
            - target_type: a `CityscapesTargetType` indicating which target to use.
        """
        # initialize preprocess transforms
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        preprocesses = [
            transforms.Resize(img_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
        compose = transforms.Compose(preprocesses)

        # initialize dataset
        super().__init__(root, split.value, target_type="color", transforms=compose)
        self.images *= repeat
        self.targets *= repeat
        self.return_file_name = return_file_name

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Union[torch.Tensor, tuple[torch.Tensor, str]]]:
        x, y = super().__getitem__(index)
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        x = (x - 0.5) * 2.
        x = x.clamp(-1., 1.)

        # convert y to rgb
        assert isinstance(y, torch.Tensor) and isinstance(y, torch.Tensor)
        y = y[:3, ...]
        y = (y - 0.5) * 2.
        y = y.clamp(-1., 1.)
        return (y, (x, self.images[index])) if self.return_file_name else (y, x)


class CityscapesForEval(Dataset):
    colored_dataset: _Cityscapes
    instance_dataset: _Cityscapes

    @property
    def unbatched_len(self) -> int:
        return len(self.colored_dataset)

    """Dataset for the cityscapes dataset."""
    def __init__(self, root: str, batch_size: int, /, img_size: Union[int, tuple[int, int]], split: CityscapesSplit = CityscapesSplit.VAL, *, device: device = devices.CPU, drop_last: bool = False, num_workers: Optional[int] = None, shuffle: bool = False) -> None:
        super().__init__(batch_size, device=device, drop_last=drop_last, num_workers=num_workers, shuffle=shuffle)
        """
        Constructor

        - Parameters:
            - root: a `str` containing the path to the root directory of the dataset.
            - img_size: an `int` or a `tuple` of `int` containing the size of the image.
            - split: a `CityscapesSplit` indicating which split to use.
            - target_type: a `CityscapesTargetType` indicating which target to use.
        """
        # initialize preprocess transforms for input color map
        img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        preprocesses = [
            transforms.Resize(img_size),
            transforms.ToTensor()
        ]
        compose = transforms.Compose(preprocesses)

        # inialized preprocess transforms for target sementic map
        target_preprocesses = [
            transforms.ToTensor(),
            CityscapesMappingLabel(),
        ]
        target_compose = transforms.Compose(target_preprocesses)

        # initialize dataset
        self.colored_dataset = _Cityscapes(root, split.value, target_type="color", transforms=compose)
        self.instance_dataset = _Cityscapes(root, split.value, target_transform=target_compose)
        assert len(self.colored_dataset) == len(self.instance_dataset), "The length of the colored dataset and the instance dataset should be the same."

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, x = self.colored_dataset[index]
        _, y = self.instance_dataset[index]
        assert isinstance(x, torch.Tensor), "The colored map is not a valid `torch.Tensor`."
        assert isinstance(y, torch.Tensor), "The instance map is not a valid `torch.Tensor`."
        return x[:3, ...], y
