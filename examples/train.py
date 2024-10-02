from torchmanager import callbacks, losses
from torchmanager_core import argparse, torch, view
from torchmanager_core.typing import Optional, Union

import data
from sde_bbdm import networks, SDEBBDMManager as Manager
from sde_bbdm.configs.train import SDEBBDMTrainingConfigs
from vqgan.networks import VQGAN


class TrainingConfigs(SDEBBDMTrainingConfigs):
    dataset: data.PairedDatasets

    def format_arguments(self) -> None:
        super().format_arguments()
        self.dataset = data.PairedDatasets(self.dataset)

    @staticmethod
    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        parser.add_argument("dataset", type=str, help="The dataset to train.")
        return SDEBBDMTrainingConfigs.get_arguments(parser)

    def show_settings(self) -> None:
        view.logger.info(f"Dataset {self.dataset}")
        super().show_settings()


def train(cfgs: TrainingConfigs, /) -> torch.nn.Module:
    # load datasets
    training_dataset, validation_dataset, _ = cfgs.dataset.load(cfgs.data_dir, cfgs.batch_size, device=cfgs.device)

    # load vqgan
    if cfgs.vqgan_path is not None:
        vqgan: VQGAN = torch.load(cfgs.vqgan_path)
        encoder = torch.nn.Sequential(vqgan.encoder, vqgan.quant_conv)
        decoder = torch.nn.Sequential(vqgan.quantizer, vqgan.post_quant_conv,vqgan.decoder)
    else:
        encoder = None
        decoder = None

    # check if resume from checkpoint
    if cfgs.ckpt_path is not None:
        # load manager directly from checkpoint
        manager: Manager[networks.UNet, Optional[torch.nn.Module], Optional[torch.nn.Module]] = Manager.from_checkpoint(cfgs.ckpt_path)  # type: ignore
        assert isinstance(manager, Manager), "Manager is not a valid `BBDMSDELinerRefinedWithLambdaV2With1000`."
        manager.encoder = encoder if encoder is not None else manager.encoder
        manager.decoder = decoder if decoder is not None else manager.decoder
        model = manager.raw_model

        # check if given c_lambda is the same as checkpoint
        if manager.c_lambda != cfgs.c_lambda:
            view.warnings.warn(f"The c_lambda saved in checkpoint is {manager.c_lambda}, given one ({cfgs.c_lambda}) will be used.", RuntimeWarning)
            manager.c_lambda = cfgs.c_lambda
    else:
        # load model
        model = networks.build_unet(3, 3)

        # load optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = losses.MAE()

        # compile manager
        manager = Manager(model, cfgs.time_steps, optimizer=optimizer, loss_fn=loss_fn, c_lambda=cfgs.c_lambda, encoder=encoder, decoder=decoder)

    # initialize callbacks
    if callbacks.Experiment is NotImplemented:
        view.warnings.warn("TensorBoard is not installed, no TensorBoard callback will be used.", RuntimeWarning)
        callbacks_list: list[callbacks.Callback] = [callbacks.LastCheckpoint(manager, cfgs.output_model)]
    else:
        experiment_callback = callbacks.Experiment(cfgs.experiment, manager, monitors={"loss": callbacks.MonitorType.MIN})
        callbacks_list: list[callbacks.Callback] = [experiment_callback]

    # train model
    model = manager.fit(training_dataset, epochs=cfgs.epochs, val_dataset=validation_dataset, callbacks_list=callbacks_list, device=cfgs.device, use_multi_gpus=cfgs.use_multi_gpus, show_verbose=cfgs.show_verbose)
    assert isinstance(model, torch.nn.Module)

    # save final model
    torch.save(model, cfgs.output_model)
    return model


if __name__ == "__main__":
    configs = TrainingConfigs.from_arguments()
    assert isinstance(configs, TrainingConfigs), "Configs is not a valid `TrainingConfigs`."
    train(configs)
