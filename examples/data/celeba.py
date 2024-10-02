import torchvision
from PIL import Image
from torchmanager.data import Dataset
from torchmanager_core import os, devices, torch
from torchmanager_core.typing import Any, Callable, Optional, Union
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms

from .split import DatasetSplit as CelebASplit


class CelebAMaskHQ(Dataset):
    """
    Dataset for the edges2shoes dataset.

    - Properties:
        - image_a_paths: a `list` of `str` containing the paths to the images in domain a.
        - image_b_paths: a `list` of `str` containing the paths to the images in domain b.
        - image_size: a `tuple` of `int` containing the size of the images.
        - return_file_name: a `bool` indicating whether to return the file name or not.
        - root_dir: a `str` containing the path to the root directory of the dataset.
        - transform: a `torchvision.transforms.Compose` to transform the images and conditions.
    """
    image_a_paths: list[str]
    image_b_path: list[str]
    image_size: tuple[int, int]
    return_file_name: bool
    root_dir: str
    split: CelebASplit
    transform: transforms.Compose

    @property
    def unbatched_len(self) -> int:
        return len(self.image_a_paths)

    def __init__(self, path: str, batch_size: int, /, img_size: Union[int, tuple[int, int]], *, device: Optional[torch.device] = None, drop_last: bool = False, num_workers: Optional[int] = None, repeat: int = 1, return_file_name: bool = False, shuffle: bool = False, split: CelebASplit = CelebASplit.TRAIN):
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
            - split: an `Face2ComicsSplit` indicating which split to use.
        """
        device = devices.CPU if device is None else device
        super().__init__(batch_size, device=device, drop_last=drop_last, num_workers=num_workers, shuffle=shuffle)
        self.return_file_name = return_file_name

        # initialize
        self.image_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.root_dir = os.path.join(path, split.value)
        self.split = split

        # search for images in domain a in folder of root_dir
        self.image_a_paths = [p for p in os.listdir(os.path.join(self.root_dir, "A")) if os.path.isfile(os.path.join(self.root_dir, "A", p))]
        self.image_a_paths *= repeat
        # search for images in domain b in folder of root_dir
        self.image_b_paths = [p for p in os.listdir(os.path.join(self.root_dir, "B")) if os.path.isfile(os.path.join(self.root_dir, "B", p))]
        self.image_b_paths *= repeat

        # assert that both domains have the same number of images
        assert len(self.image_a_paths) == len(self.image_b_paths), "Both domains must have the same number of images."

        # TODO
        # self.image_a_paths = self.image_a_paths[:500]
        # self.image_b_paths = self.image_b_paths[:500]

        # initialize transforms
        transforms_list: list[Callable[..., tuple[Any, Any]]] = [transforms.RandomHorizontalFlip(p=0)] if split == CelebASplit.TRAIN else []
        transforms_list.extend([
            transforms.Resize(self.image_size),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        self.transform = transforms.Compose(transforms_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Union[torch.Tensor, tuple[torch.Tensor, str]]]:
        # load image in domain a
        img_a_file = self.image_a_paths[index]
        img_a_path = os.path.join(self.root_dir, "A",img_a_file)
        image_a = Image.open(img_a_path)
        
        # load image in domain b
        img_b_file = self.image_b_paths[index]
        img_b_path = os.path.join(self.root_dir, "B",img_b_file)
        image_b = Image.open(img_b_path)

        # convert to RGB if necessary
        if not image_a.mode == 'RGB':
            image_a = image_a.convert('RGB')

        # convert to RGB if necessary
        if not image_b.mode == 'RGB':
            image_b = image_b.convert('RGB')

        # apply transform
        image_a, image_b = self.transform(image_a, image_b)
        assert isinstance(image_a, torch.Tensor), 'Image in domain A is not a valid `torch.tensor`.'
        assert isinstance(image_b, torch.Tensor), 'Image in domain B is not a valid `torch.tensor`.'

        # normalize image
        image_a = (image_a - 0.5) * 2.
        image_a.clamp(-1., 1.)

        # normalize condition
        image_b = (image_b - 0.5) * 2.
        image_b.clamp(-1., 1.)
        return (image_a, (image_b, img_a_file)) if self.return_file_name else (image_a, image_b)
