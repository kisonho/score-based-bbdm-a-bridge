import argparse, os, torch

import data


class TestingConfig(argparse.Namespace):
    batch_size: int
    dataset_dir: str
    device: torch.device
    model: str
    show_verbose: bool
    # target: data.MoNuSegTarget
    use_multi_gpus: bool

    @classmethod
    def from_arguments(cls, *arguments: str):
        # initialize arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset_dir", type=str, help="The directory of dataset.")
        parser.add_argument("model", type=str, help="The model to test.")

        # add testing argumetns
        testing_args = parser.add_argument_group("Testing Arguments")
        testing_args.add_argument("-b", "--batch_size", type=int, default=6, help="The training batch size, default is 6.")
        testing_args.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show training progress bar.")
        # testing_args.add_argument("--target", type=str, default="image", help="The target data of VQGAN, default is 'image'.")

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        device_args.add_argument("--device", type=str, default="cuda", help="Device for training, default is \'cuda\'.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi GPUs.")

        # parse arguments
        if len(arguments) > 0:
            configs = parser.parse_args(arguments, namespace=cls())
        else:
            configs = parser.parse_args(namespace=cls())

        # format arguments
        configs.format_arguments()
        return configs

    def format_arguments(self) -> None:
        self.dataset_dir = os.path.normpath(self.dataset_dir)
        self.device = torch.device(self.device)
        self.model = os.path.normpath(self.model)
        # self.target = data.MoNuSegTarget(self.target)
        assert self.batch_size > 0, f"Batch size must be a positive number, got {self.batch_size}."
