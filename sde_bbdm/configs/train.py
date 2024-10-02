import argparse, diffusion, os
from diffusion.configs import TrainingConfigs
from torchmanager_core import view
from typing import Optional, Union

from sde_bbdm.version import DESCRIPTION


class SDEBBDMTrainingConfigs(TrainingConfigs):
    c_lambda: float
    vqgan_path: Optional[str]

    def format_arguments(self) -> None:
        super().format_arguments()
        assert self.c_lambda > 0, "Lambda must be a positive number."
        self.vqgan_path = os.path.normpath(self.vqgan_path) if self.vqgan_path is not None else None

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        parser = TrainingConfigs.get_arguments(parser)
        sde_bbdm_args = parser.add_argument_group("Score Based BBDM Arguments")
        sde_bbdm_args.add_argument("-l", "--c_lambda", type=float, default=2, help="The lambda for the loss function, default is 2.")
        sde_bbdm_args.add_argument("-vq", "--vqgan_path", type=str, default=None, help="The path for the VQGAN model if using latent space.")
        return parser

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"diffusion={diffusion.VERSION}")

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"Score Based BBDM settings: c_lambda={self.c_lambda}, vqgan_path={self.vqgan_path}")
