from diffusion import metrics
from torchmanager_core import argparse, devices, os, torch, view
from torchmanager_core.typing import Optional
from torchvision import transforms

import data, vqgan
from sde_bbdm import SDEBBDMManager as Manager
from sde_bbdm.configs import SDEBBDMEvalConfigs


class EvalMIoUConfig(SDEBBDMEvalConfigs):
    seg_model_dir: str

    def format_arguments(self) -> None:
        self.seg_model_dir = os.path.normpath(self.seg_model_dir)
        super().format_arguments()

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("seg_model_dir", type=str, help="The segmentation model directory.")
        parser = SDEBBDMEvalConfigs.get_arguments(parser)  # type: ignore
        return parser

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"Target directory: {self.seg_model_dir}")


def eval(configs: EvalMIoUConfig, /, model: Optional[torch.nn.Module] = None) -> dict[str, float]:
    """
    Test with `diffusion.configs.TestingConfigs`

    - Parameters:
        - model: An optional pre-trained `torch.nn.Module`
        - configs: A `diffusion.configs.TestingConfigs` for testing
    - Returns: A `dict` of results with name as `str` and value as `float`
    """
    # load dataset
    num_workers = os.cpu_count()
    num_workers = 0 if num_workers is None else num_workers
    _, testing_dataset = data.PairedDatasets.CITYSCAPES.load(configs.data_dir, configs.batch_size, device=configs.device)

    # load segmentation model
    seg_model = torch.load(configs.seg_model_dir)
    assert isinstance(seg_model, torch.nn.Module), "The pre-trained model is not a valid PyTorch model."
    seg_model = seg_model.eval()
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # initialize metrics
    metric_fns: dict[str, metrics.Metric] = {
        "mIoU": metrics.MIoU(seg_model, normalize_fn=normalize_fn),
    }

    # load checkpoint
    if configs.model is not None and configs.model.endswith(".model"):
        # load checkpoint
        manager: Manager[torch.nn.Module] = Manager.from_checkpoint(configs.model, map_location=devices.CPU)  # type: ignore
        manager.reset()

        # set metrics
        manager.loss_fn = None
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

    # create fast sampling steps
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
    configs = EvalMIoUConfig.from_arguments()
    assert isinstance(configs, EvalMIoUConfig), "Configs is not a valid `EvalMIoUConfig`."
    result = eval(configs)
    view.logger.info(result)
