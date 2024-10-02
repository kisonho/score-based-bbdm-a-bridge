from torchmanager.configs import Configs
from torchmanager_core import argparse, devices, os, torch, view
from torchmanager_core.typing import Union

import data


class TrainingConfigs(Configs):
    batch_size: int
    beta: float
    dataset_dir: str
    device: torch.device
    disc_start: int
    epochs: int
    latent_dim: int
    latent_scale: int
    learning_rate: float
    num_codebook_vectors: int
    output_model: str
    show_verbose: bool
    # target: data.MoNuSegTarget
    use_multi_gpus: bool

    def format_arguments(self) -> None:
        # format data
        super().format_arguments()
        self.dataset_dir = os.path.normpath(self.dataset_dir)
        self.device = torch.device(self.device)
        self.output_model = os.path.normpath(self.output_model)
        # self.target = data.MoNuSegTarget(self.target)

        # format logger
        formatter = view.logging.Formatter("%(message)s")
        console = view.logging.StreamHandler()
        console.setLevel(view.logging.INFO)
        console.setFormatter(formatter)
        view.logger.addHandler(console)

        # assert parameters
        assert self.batch_size > 0, f"Batch size must be a positive number, got {self.batch_size}."
        assert self.beta > 0, f"The beta coefficient must be a positive number, got {self.beta}."
        assert self.disc_start >= 0, f"The discrimination starting step must be a non-negative number, got {self.disc_start}."
        assert self.epochs > 0, f"Batch size must be a positive number, got {self.epochs}."
        assert self.latent_dim > 0, f"Latent dimension must be a positive number, got {self.latent_dim}."
        assert self.latent_scale > 0, f"Latent scale must be a positive number, got {self.latent_scale}."
        assert self.learning_rate > 0, f"Learning rate must be a positive number, got {self.learning_rate}."
        assert self.num_codebook_vectors > 0, f"The number of codebook vectors must be a positive number, got {self.num_codebook_vectors}."
        if self.device.type == "cuda":
            assert devices.GPU is not NotImplemented, "CUDA device is not available."

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        # required arguments
        parser.add_argument("dataset_dir", type=str, help="The directory of dataset.")
        parser.add_argument("output_model", type=str, help="The output of trained model.")

        # training arguments
        training_args = parser.add_argument_group("Training Arguments")
        training_args.add_argument("-e", "--epochs", type=int, default=100, help="The number of training epochs, default is 100.")
        training_args.add_argument("-b", "--batch_size", type=int, default=6, help="The training batch size, default is 6.")
        training_args.add_argument("-lr", "--learning_rate", type=float, default=4.5e-6, help='Learning rate (default: 4.5e-6)')
        training_args.add_argument("--beta", type=float, default=0.25, help="The beta of commitment loss in VQGAN.")
        training_args.add_argument("--disc_start", type=int, default=10000, help="Discriminator starting step, default is 10000.")
        training_args.add_argument("-f", "--latent_scale", type=int, default=4, help="The scale of latent dimension, or `f`, default is 4.")
        training_args.add_argument("-z", "--num_codebook_vectors", type=int, default=256, help="The number of latent vectors, or `|Z|`, default is 256.")
        training_args.add_argument("-c", "--latent_dim", type=int, default=3, help="The latent dimension, or `c`, default is 3.")
        # training_args.add_argument("--target", type=str, default="image", help="The target data of VQGAN, default is 'image'.")
        training_args = Configs.get_arguments(training_args)
        training_args.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show training progress bar.")

        # device arguments
        device_args = parser.add_argument_group("Device Arguments")
        device_args.add_argument("--device", type=str, default="cuda", help="Device for training, default is \'cuda\'.")
        device_args.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi GPUs.")
        return parser

    def show_environments(self) -> None:
        super().show_environments()

    def show_settings(self) -> None:
        view.logger.info(f"Dataset directory: {self.dataset_dir}")
        view.logger.info(f"Output model directory: {self.output_model}")
        view.logger.info(f"Training details: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}, beta={self.beta}, disc_start={self.disc_start}")
        view.logger.info(f"VQGAN network settings: f={self.latent_scale}, c={self.latent_dim}, |Z|={self.num_codebook_vectors}")
        view.logger.info(f"Device details: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
