import ssl
from diffusion import metrics
from torchmanager_core import argparse, devices, torch, view
from torchmanager_core.typing import Optional, Union
from torchvision import models

import data, vqgan
from sde_bbdm import SDEBBDMManager as Manager
from sde_bbdm.configs import SDEBBDMEvalConfigs


class EvalConfigs(SDEBBDMEvalConfigs):
    dataset: data.PairedDatasets
    repeat: int

    def format_arguments(self) -> None:
        super().format_arguments()
        self.dataset = data.PairedDatasets(self.dataset)
        assert self.repeat > 0, "Repeat must be a positive number."

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        parser.add_argument("dataset", type=str, help="The dataset directory.")
        parser.add_argument("-r", "--repeat", type=int, default=1, help="The repeat of the dataset, default is 1.")
        return SDEBBDMEvalConfigs.get_arguments(parser)

    def show_settings(self) -> None:
        view.logger.info(f"Dataset {self.dataset}: repeat={self.repeat}")
        super().show_settings()


def eval(configs: EvalConfigs, /, model: Optional[torch.nn.Module] = None) -> dict[str, float]:
    """
    Test with `diffusion.configs.TestingConfigs`

    - Parameters:
        - model: An optional pre-trained `torch.nn.Module`
        - configs: A `diffusion.configs.TestingConfigs` for testing
    - Returns: A `dict` of results with name as `str` and value as `float`
    """
    # initialize FID
    ssl._create_default_https_context = ssl._create_unverified_context
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()  # type: ignore
    inception = torch.nn.Sequential(torch.nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False), inception)
    inception.eval()
    fid = metrics.FID(inception)

    # initialize LPIPS
    lpips = metrics.LPIPS()

    # initialize metrics
    metric_fns = {
        "FID": fid,
        "LPIPS": lpips,
    }

    # load checkpoint
    if configs.model is not None and configs.model.endswith(".model"):
        # load checkpoint
        manager: Manager[torch.nn.Module] = Manager.from_checkpoint(configs.model, map_location=devices.CPU)  # type: ignore
        manager.reset()

        # set metrics
        manager.metric_fns = metric_fns
    else:
        # load beta space
        assert model is not None or configs.model is not None, "Either pre-trained model should be given as parameters or its path has been given in configurations."
        model = torch.load(configs.model, map_location=devices.CPU) if configs.model is not None else model
        assert isinstance(model, torch.nn.Module), "The pre-trained model is not a valid PyTorch model or torchmanager checkpoint."
        assert configs.time_steps is not None, "Time steps is required when loading a PyTorch model."
        manager = Manager(model, configs.time_steps, metrics=metric_fns)  # type: ignore

    # set time steps
    if configs.time_steps is not None:
        manager.time_steps = configs.time_steps

    # set lambda
    if configs.c_lambda is not None:
        manager.c_lambda = configs.c_lambda

    # set encoder and decoder
    if configs.vqgan_path is not None:
        autoencoder = torch.load(configs.vqgan_path, map_location=devices.CPU)
        assert isinstance(autoencoder, vqgan.VQGAN), "The pre-trained VQGAN model is not a valid `vqgan.VQGAN`."
        encoder = torch.nn.Sequential(autoencoder.encoder, autoencoder.quant_conv)
        decoder = torch.nn.Sequential(autoencoder.quantizer, autoencoder.post_quant_conv,autoencoder.decoder)
        manager.encoder = encoder
        manager.decoder = decoder

    # load dataset
    _, _, testing_dataset = configs.dataset.load(configs.data_dir, configs.batch_size, repeat=configs.repeat, device=configs.device)

    # # create fast sampling steps
    num_timesteps = 1000 if configs.time_steps is None else configs.time_steps

    # case1
    midsteps1: list[int] = torch.arange(num_timesteps, 904, step=-2).long().numpy().tolist()
    midsteps2: list[int] = torch.arange(902, 1, step=-6).long().numpy().tolist()
    steps = midsteps1 + midsteps2 + [1]

    # combine two lists
    steps = midsteps1 + midsteps2

    # evaluation
    result = manager.test(testing_dataset, sampling_images=True, device=configs.device, use_multi_gpus=configs.use_multi_gpus, show_verbose=configs.show_verbose, sampling_range=steps, fast_sampling=True)
    return result


if __name__ == "__main__":
    configs = EvalConfigs.from_arguments()
    assert isinstance(configs, EvalConfigs), "Configs is not a valid `SDEBBDMEvalConfigs`."
    result = eval(configs)
    view.logger.info(result)
