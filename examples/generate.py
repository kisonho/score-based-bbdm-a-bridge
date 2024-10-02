from torch.utils.data import DataLoader
from torchmanager_core import argparse, devices, os, torch, view
from torchmanager_core.typing import Optional
from torchvision.utils import save_image

import data, vqgan
from sde_bbdm import SDEBBDMManager as Manager
from sde_bbdm.configs import SDEBBDMEvalConfigs


class GeneratingConfigs(SDEBBDMEvalConfigs):
    dataset: data.PairedDatasets
    output_dir: str

    def format_arguments(self) -> None:
        self.dataset = data.PairedDatasets(self.dataset)
        self.output_dir = os.path.normpath(self.output_dir)
        super().format_arguments()
   
    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("dataset", type=str, help="The dataset to generate images from.")
        parser.add_argument("output_dir", type=str, help="The target directory.")
        parser = SDEBBDMEvalConfigs.get_arguments(parser)  # type: ignore
        return parser

    def show_settings(self) -> None:
        view.logger.info(f"Dataset: {self.dataset.name}")
        super().show_settings()
        view.logger.info(f"Target directory: {self.output_dir}")


def generate(configs: GeneratingConfigs, /, model: Optional[torch.nn.Module] = None) -> list[tuple[torch.Tensor, str]]:
    """
    Generate images from a pre-trained model.

    - Parameters:
        - model: An optional pre-trained `torch.nn.Module`
        - configs: A `diffusion.configs.TestingConfigs` for testing
    - Returns: A `list` of generated images in `torch.Tensor`.
    """
    # load dataset
    testing_dataset = configs.dataset.load_for_generation(configs.data_dir, configs.batch_size, repeat=1, device=configs.device)

    # load checkpoint
    if configs.model is not None and configs.model.endswith(".model"):
        # load checkpoint
        manager: Manager[torch.nn.Module] = Manager.from_checkpoint(configs.model, map_location=devices.CPU)  # type: ignore
        manager.reset()
    else:
        # load beta space
        assert model is not None or configs.model is not None, "Either pre-trained model should be given as parameters or its path has been given in configurations."
        model = torch.load(configs.model, map_location=devices.CPU) if configs.model is not None else model
        assert isinstance(model, torch.nn.Module), "The pre-trained model is not a valid PyTorch model or torchmanager checkpoint."
        assert configs.time_steps is not None, "Time steps is required when loading a PyTorch model."
        manager = Manager(model, configs.time_steps)

    # set time steps
    if configs.time_steps is not None:
        manager.time_steps = configs.time_steps

    # set lambda
    if configs.c_lambda is not None:
        manager.c_lambda = configs.c_lambda

    # set encoder and decoder
    if configs.vqgan_path is not None:
        autoencoder: vqgan.VQGAN = torch.load(configs.vqgan_path, map_location=devices.CPU)
        encoder = torch.nn.Sequential(autoencoder.encoder, autoencoder.quant_conv)
        decoder = torch.nn.Sequential(autoencoder.quantizer, autoencoder.post_quant_conv,autoencoder.decoder)
        manager.encoder = encoder
        manager.decoder = decoder

    # create fast sampling steps
    if configs.fast_sampling:
        num_timesteps = 1000 if configs.time_steps is None else configs.time_steps
        midsteps1: list[int] = torch.arange(num_timesteps, 904, step=-2).long().numpy().tolist()
        midsteps2: list[int] = torch.arange(902, 1, step=-6).long().numpy().tolist()
        steps = midsteps1 + midsteps2 + [1]
        steps = midsteps1 + midsteps2
    else:
        steps = None

    # initialize generation
    generated_imgs: list[tuple[torch.Tensor, str]] = []
    view.logger.info("Generating images...")
    dataset_len = len(testing_dataset) if isinstance(testing_dataset, DataLoader) else testing_dataset.batched_len
    progress_bar = view.tqdm(total=dataset_len) if configs.show_verbose else None

    # generate
    try:
        for images, label in testing_dataset:
            assert len(label) == 2, "The dataset should return a label in tuple of targets and file name."
            _, name = label
            assert isinstance(images, torch.Tensor), "Input images is not a valid `torch.Tensor`."
            images = manager.predict(images.shape[0], 256, condition=images, fast_sampling=configs.fast_sampling, sampling_range=steps, device=configs.device, use_multi_gpus=configs.use_multi_gpus)
            generated_imgs.extend(zip(images, name))
            if progress_bar is not None:
                progress_bar.update()
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return generated_imgs


def save(images: list[tuple[torch.Tensor, str]], target_directory: str, *, show_verbose: bool = False, starting_index: int = 0) -> None:
    """
    Save a list of generated images to a target directory with names as indices.

    Parameters:
        - images: A `list` of generated images in `torch.Tensor`.
        - target_directory: Target directory in `str` to save the images.
        - show_verbose: A `bool` indicating whether to show verbose or not.
        - starting_index: The starting index in `int` for naming the images.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)
    view.logger.info(f"Saving images to `{target_directory}`...")
    progress_bar = view.tqdm(total=len(images)) if show_verbose else None

    # Save each image with its index as the filename
    try:
        for image, name in images:
            image = image / 2
            image += 0.5
            image = image.clip(0,1)
            filename = os.path.join(target_directory, name.replace("jpg", "png"))
            save_image(image, filename)
            if progress_bar is not None:
                progress_bar.update()
    finally:
        if progress_bar is not None:
            progress_bar.close()


if __name__ == "__main__":
    # get configs
    configs = GeneratingConfigs.from_arguments()
    assert isinstance(configs, GeneratingConfigs), "Configs is not a valid `GeneratingConfigs`."

    # generate
    generated_images = generate(configs)
    save(generated_images, configs.output_dir, show_verbose=configs.show_verbose)
