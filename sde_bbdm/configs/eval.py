import argparse, diffusion, torchmanager
from torchmanager.configs import Configs as _Configs
from torchmanager_core import argparse, os, torch, view, _raise
from torchmanager_core.typing import Optional, Union

from sde_bbdm.version import DESCRIPTION


class SDEBBDMEvalConfigs(_Configs):
    """Evaluation Configurations"""
    batch_size: int
    c_lambda: Optional[float]
    data_dir: str
    device: Optional[torch.device]
    fast_sampling: bool
    model: str
    show_verbose: bool
    time_steps: Optional[int]
    use_multi_gpus: bool
    vqgan_path: Optional[str]

    def format_arguments(self) -> None:
        # format arguments
        super().format_arguments()
        self.data_dir = os.path.normpath(self.data_dir)
        self.device = torch.device(self.device) if self.device is not None else None
        self.model = os.path.normpath(self.model)
        self.vqgan_path = os.path.normpath(self.vqgan_path) if self.vqgan_path is not None else None

        # assert formats
        assert self.batch_size > 0, _raise(ValueError(f"Batch size must be a positive number, got {self.batch_size}."))
        if self.c_lambda is not None:
            assert self.c_lambda > 0, "Lambda must be a positive number."
        if self.time_steps is not None:
            assert self.time_steps > 0, _raise(ValueError(f"Time steps must be a positive number, got {self.time_steps}."))
        
        # format logging
        formatter = view.logging.Formatter("%(message)s")
        console = view.logging.StreamHandler()
        console.setLevel(view.logging.INFO)
        console.setFormatter(formatter)
        view.logger.addHandler(console)

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # experiment arguments
        parser.add_argument("data_dir", type=str, help="The dataset directory.")
        parser.add_argument("model", type=str, help="The path for a pre-trained PyTorch model or a torchmanager checkpoint, default is `None`.")

        # training arguments
        testing_args = parser.add_argument_group("Testing Arguments")
        testing_args.add_argument("-b", "--batch_size", type=int, default=64, help="The batch size, default is 64.")
        testing_args.add_argument("--fast_sampling", action="store_true", default=False, help="A flag to use fast sampling.")
        testing_args.add_argument("--show_verbose", action="store_true", default=False, help="A flag to show verbose.")
        testing_args.add_argument("-t", "--time_steps", type=int, default=None, help="The total time steps of diffusion model, default is `None` (Checkpoint is needed).")
        testing_args = _Configs.get_arguments(testing_args)

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        device_args.add_argument("--device", type=str, default=None, help="The target device to run for the experiment.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="A flag to use multiple GPUs during training.")

        # sde BBDM arguments
        sde_bbdm_args = parser.add_argument_group("SDE BBDM Arguments")
        sde_bbdm_args.add_argument("-l", "--c_lambda", type=float, default=None, help="The lambda for the loss function, default is `None`.")
        sde_bbdm_args.add_argument("-vq", "--vqgan_path", type=str, default=None, help="The path for the VQGAN model.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager={torchmanager.version}")
        view.logger.info(f"diffusion={diffusion.VERSION}")

    def show_settings(self) -> None:
        view.logger.info(f"Data directory: {self.data_dir}")
        view.logger.info(f"Pre-trained model: {self.model}")
        view.logger.info(f"Testing settings: batch_size={self.batch_size}, show_verbose={self.show_verbose}")
        view.logger.info(f"Diffusion model settings: fast_sampling={self.fast_sampling}, time_steps={self.time_steps}")
        view.logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
        view.logger.info(f"SDE BBDM settings: c_lambda={self.c_lambda}, vqgan_path={self.vqgan_path}")
