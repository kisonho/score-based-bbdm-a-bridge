from torch.utils.data import DataLoader
from torchmanager.data import Dataset
from torchmanager_core import os, torch
from torchmanager_core.typing import Enum, Optional, Union

from .celeba import CelebAMaskHQ
from .cityscapes import Cityscapes
from .edge2shoes import Edge2Shoes
from .face2comics import Face2Comics
from .split import DatasetSplit


class PairedDatasets(Enum):
    """Enum for the paired datasets."""
    CELEBA = "celeba"
    """The celeba dataset."""
    CITYSCAPES = "cityscapes"
    """The cityscapes dataset."""
    EDGE2SHOES = "edge2shoes"
    """The edge2shoes dataset."""
    FACE2COMICS = "face2comics"
    """The face2comics dataset."""

    def load(self, root_dir: str, /, batch_size: int, *, repeat: int = 1, device: Optional[torch.device] = None) -> tuple[Union[DataLoader, Dataset], ...]:
        """
        Load the dataset.

        - Parameters:
            - root_dir: The root directory of the dataset in `str`.
            - batch_size: The batch size in `int`.
            - repeat: The repeat of the testing dataset in `int`, default is 1.
            - device: The device to load the data on in `torch.device`.
        - Returns: A `tuple` of the training, validation, and testing dataset in `torchmanager.data.Dataset` or `torch.utils.data.DataLoader`.
        """
        # check supported datasets
        if self == PairedDatasets.CELEBA:
            training_dataset = CelebAMaskHQ(root_dir, batch_size, img_size=256, device=device, drop_last=True, num_workers=os.cpu_count(), shuffle=True, split=DatasetSplit.TRAIN)
            validation_dataset = testing_dataset = CelebAMaskHQ(root_dir, batch_size, img_size=256, device=device, num_workers=os.cpu_count(), repeat=repeat, split=DatasetSplit.VAL)
        elif self == PairedDatasets.CITYSCAPES:
            # initialize device
            pin_memory = True if device is not None and device.type != "cpu" else False
            pin_memory_device = str(device) if device is not None else ""

            # load datasets
            training_dataset = Cityscapes(root_dir, 256, split=DatasetSplit.TRAIN)
            training_dataset = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory, pin_memory_device=pin_memory_device)
            validation_dataset = Cityscapes(root_dir, 256, split=DatasetSplit.VAL, repeat=repeat)
            validation_dataset = testing_dataset = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=pin_memory, pin_memory_device=pin_memory_device)
        elif self == PairedDatasets.EDGE2SHOES:
            training_dataset = Edge2Shoes(root_dir, batch_size, img_size=256, device=device, drop_last=True, num_workers=os.cpu_count(), shuffle=True, split=DatasetSplit.TRAIN)
            validation_dataset = testing_dataset = Edge2Shoes(root_dir, batch_size, img_size=256, device=device, num_workers=os.cpu_count(), repeat=repeat, split=DatasetSplit.VAL)
        elif self == PairedDatasets.FACE2COMICS:
            training_dataset = Face2Comics(root_dir, batch_size, img_size=256, device=device, drop_last=True, num_workers=os.cpu_count(), shuffle=True, split=DatasetSplit.TRAIN)
            validation_dataset = testing_dataset = Face2Comics(root_dir, batch_size, img_size=256, device=device, num_workers=os.cpu_count(), repeat=repeat, split=DatasetSplit.VAL)
        else:
            raise NotImplementedError(f"Paired dataset `{self.name}` is not supported.")
        return training_dataset, validation_dataset, testing_dataset

    def load_for_generation(self, root_dir: str, /, batch_size: int, *, repeat: int = 1, device: Optional[torch.device] = None) -> Union[DataLoader, Dataset]:
        """
        Load dataset for generation.

        - Parameters:
            - root_dir: The root directory of the dataset in `str`.
            - batch_size: The batch size in `int`.
            - repeat: The repeat of the testing dataset in `int`, default is 1.
            - device: The device to load the data on in `torch.device`.
        - Returns: A `tuple` of the training, validation, and testing dataset in `torchmanager.data.Dataset` or `torch.utils.data.DataLoader`.
        """
        # check supported datasets
        if self == PairedDatasets.CELEBA:
            validation_dataset = CelebAMaskHQ(root_dir, batch_size, img_size=256, device=device, num_workers=os.cpu_count(), repeat=repeat, return_file_name=True, split=DatasetSplit.VAL)
        elif self == PairedDatasets.CITYSCAPES:
            # initialize device
            pin_memory = True if device is not None and device.type != "cpu" else False
            pin_memory_device = str(device) if device is not None else ""

            # load datasets
            validation_dataset = Cityscapes(root_dir, 256, split=DatasetSplit.VAL, repeat=repeat, return_file_name=True)
            validation_dataset = DataLoader(validation_dataset, batch_size=batch_size, pin_memory=pin_memory, pin_memory_device=pin_memory_device)
        elif self == PairedDatasets.EDGE2SHOES:
            validation_dataset = Edge2Shoes(root_dir, batch_size, img_size=256, device=device, num_workers=os.cpu_count(), repeat=repeat, return_file_name=True, split=DatasetSplit.VAL)
        elif self == PairedDatasets.FACE2COMICS:
            validation_dataset = Face2Comics(root_dir, batch_size, img_size=256, device=device, num_workers=os.cpu_count(), repeat=repeat, return_file_name=True, split=DatasetSplit.VAL)
        else:
            raise NotImplementedError(f"Paired dataset `{self.name}` is not supported.")
        return validation_dataset
