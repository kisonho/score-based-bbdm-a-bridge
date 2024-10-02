"""
Edges2Shoes dataset in torchmanager dataset.

Code modified from https://github.com/xuekt98/BBDM
"""
import numpy as np, torchvision
from PIL import Image
from torchmanager.data import Dataset
from torchmanager_core import os, devices, torch
from torchmanager_core.typing import Any, Callable, Optional, Union
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms

from .split import DatasetSplit as Edge2ShoesSplit


class Edge2Shoes(Dataset):
    """
    Dataset for the edges2shoes dataset.

    - Properties:
        - image_paths: a `list` of `str` containing the paths to the images.
        - image_size: a `tuple` of `int` containing the size of the images.
        - root_dir: a `str` containing the path to the root directory of the dataset.
        - transform: a `torchvision.transforms.Compose` to transform the images and conditions.
    """
    image_paths: list[str]
    image_size: tuple[int, int]
    return_file_name: bool
    root_dir: str
    transform: transforms.Compose

    def __init__(self, path: str, batch_size: int, /, img_size: Union[int, tuple[int, int]], *, device: Optional[torch.device] = None, drop_last: bool = False, num_workers: Optional[int] = None, repeat: int = 1, return_file_name: bool = False, shuffle: bool = False, split: Edge2ShoesSplit = Edge2ShoesSplit.TRAIN):
        """
        Constructor

        - Parameters:
            - path: a `str` containing the path to the root directory of the dataset.
            - batch_size: an `int` containing the batch size.
            - img_size: an `int` or a `tuple` of `int` containing the size of the images.
            - device: a `torch.device` to load the data on.
            - drop_last: a `bool` indicating whether to drop the last batch or not.
            - num_workers: an `int` containing the number of workers to use for loading the data.
            - repeat: an `int` containing the number of times to repeat the dataset.
            - return_file_name: a `bool` indicating whether to return the file name or not.
            - shuffle: a `bool` indicating whether to shuffle the data or not.
            - split: an `Edge2ShoesSplit` indicating which split to use.
        """
        device = devices.CPU if device is None else device
        super().__init__(batch_size, device=device, drop_last=drop_last, num_workers=num_workers, shuffle=shuffle)
        self.return_file_name = return_file_name

        # initialize
        self.image_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.root_dir = os.path.join(path, split.value)

        # search for images in folder of root_dir
        self.image_paths = [p for p in os.listdir(os.path.join(self.root_dir)) if os.path.isfile(os.path.join(self.root_dir, p)) and p.endswith('.jpg')]
        self.image_paths *= repeat

        # TODO
        # self.image_paths = self.image_paths[:500]

        # initialize transforms
        transforms_list: list[Callable[..., tuple[Any, Any]]] = [transforms.RandomHorizontalFlip(p=0)] if split == Edge2ShoesSplit.TRAIN else []
        transforms_list.extend([
            transforms.Resize(self.image_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.transform = transforms.Compose(transforms_list)

    @property
    def unbatched_len(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Union[torch.Tensor, tuple[torch.Tensor, str]]]:
        # load image
        img_file = self.image_paths[index]
        img_path = os.path.join(self.root_dir, img_file)
        combined_image = Image.open(img_path)
        combined_image = np.array(combined_image)
        w = combined_image.shape[1]
        image = Image.fromarray(combined_image[:, int(w / 2):, :])
        condition = Image.fromarray(combined_image[:, :int(w / 2), :])

        # convert to RGB if necessary
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        # convert to RGB if necessary
        if not condition.mode == 'RGB':
            condition = condition.convert('RGB')

        # apply transform
        image, condition = self.transform(image, condition)
        assert isinstance(image, torch.Tensor), 'Image is not a valid `torch.tensor`.'
        assert isinstance(condition, torch.Tensor), 'Condition is not a valid `torch.tensor`.'

        # normalize image
        image = (image - 0.5) * 2.
        image.clamp(-1., 1.)

        # normalize condition
        condition = (condition - 0.5) * 2.
        condition.clamp(-1., 1.)
        return (condition, (image, img_file)) if self.return_file_name else (condition, image)
