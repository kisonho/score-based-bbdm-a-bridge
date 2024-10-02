from diffusion import scheduling, DiffusionData
from diffusion.managers import DDPMManager
from torch.nn.modules.module import Module
from torchmanager import losses, metrics
from torchmanager_core import devices, torch
from torchmanager_core.typing import Any, Generic, Iterable, Module, Optional, TypeVar, Union

from .latent import LatentDiffusionManager, E, D

ConditionEnc = TypeVar('ConditionEnc', bound=Optional[torch.nn.Module])


class ConditionalLDMManager(LatentDiffusionManager[Module, E, D], DDPMManager[Module], Generic[Module, E, ConditionEnc, D]):
    """
    The manager for conditional LDM

    * extends: `LatentDiffusionManager`, `DDPMManager`
    * generic: `Module`, `E`, `D`, `ConditionEnc`

    - Parameters:
        - condition_encoder: An optional condition encoder in `ConditionEnc` to encode condition
    """
    condition_encoder: Union[ConditionEnc, torch.nn.DataParallel[torch.nn.Module]]

    def __init__(self, model: Module, beta_space: scheduling.BetaSpace, time_steps: int, /, encoder: E = None, decoder: D = None, condition_encoder: ConditionEnc = None, *, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[losses.Loss, dict[str, losses.Loss]]] = None, metrics: dict[str, metrics.Metric] = {}) -> None:
        """
        Constructor

        - Prarameters:
            - model: An optional target `torch.nn.Module` to be trained
            - beta_space: A `BetaSpace` object to schedule the beta
            - time_steps: An `int` of total number of steps
            - encoder: An optional encoder in `E` to enter latent space
            - decoder: An optional decoder in `D` to exit latent space
            - condition_encoder: An optional condition encoder in `ConditionEnc` to encode condition
            - optimizer: An optional `torch.optim.Optimizer` to train the model
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
        """
        LatentDiffusionManager.__init__(self, model, time_steps, encoder=encoder, decoder=decoder, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)
        DDPMManager.__init__(self, model, beta_space, time_steps, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)

        # initialize condition encoder
        self.condition_encoder = condition_encoder

    def data_parallel(self, target_devices: list[torch.device]) -> bool:
        if self.condition_encoder is not None:
            self.condition_encoder = torch.nn.DataParallel(self.condition_encoder, target_devices)
        return super().data_parallel(target_devices)

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_noise: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # predict noise
        predicted_noise, _ = self.forward(data) if predicted_noise is None else predicted_noise
        assert isinstance(predicted_noise, torch.Tensor), "The model must return a `torch.Tensor` as predicted noise."

        if tau == 1:
            y = data.x - predicted_noise
        else:
            # select parameters corresponding to the currently considered timestep
            tau_minus_one_full = torch.full((data.x.shape[0],), tau_minus_one, dtype=torch.long, device=data.x.device)
            a_t = self.beta_space.sample_alphas(data.t, data.x.shape)
            a_prev = self.beta_space.sample_alphas(tau_minus_one_full, data.x.shape)
            sigma_t = self.beta_space.sample_posterior_variance(data.t, data.x.shape)
            sqrt_one_minus_at = self.beta_space.sample_sqrt_one_minus_alphas_cumprod(data.t, data.x.shape)

            # current prediction for x_0
            pred_x0 = (data.x - sqrt_one_minus_at * predicted_noise) / a_t.sqrt()
            y = (1. - a_prev - sigma_t**2).sqrt() * predicted_noise
            y += a_prev.sqrt() * pred_x0
            noise = torch.randn_like(data.x, device=y.device)
            y += noise * sigma_t.sqrt()
        return (y, predicted_noise) if return_noise else y

    def forward(self, input: DiffusionData, target: Optional[Any] = None) -> tuple[Any, Optional[Any]]:
        if self.condition_encoder is None:
            assert input.condition is not None, 'Condition is required for LDM.'
            x = torch.cat([input.x, input.condition], dim=1)
            input = DiffusionData(x, input.t)
        return super().forward(input, target)

    def reset(self, cpu: torch.device = devices.CPU) -> None:
        if isinstance(self.condition_encoder, torch.nn.DataParallel):
            self.condition_encoder = self.condition_encoder.module.to(cpu)  # type: ignore
        return super().reset(cpu)

    def sampling(self, num_images: int, x_t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Iterable[int], reversed, range]] = None, show_verbose: bool = False) -> list[torch.Tensor]:
        '''
        Samples a given number of images

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - condition: An optional `torch.Tensor` of the condition to generate images
            - start_index: An optional `int` of the start index of reversed time step
            - end_index: An `int` of the end index of reversed time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # enter latent space
        assert condition is not None, 'Condition is required for sampling.'
        z_condition: Union[torch.Tensor, tuple[torch.Tensor, Any]] = self.encode(condition) if self.condition_encoder is None else self.condition_encoder(condition)
        z_condition, args = (z_condition, None) if isinstance(z_condition, torch.Tensor) else z_condition
        z_t = torch.randn_like(z_condition)

        # sampling
        if fast_sampling:
            assert sampling_range is not None, 'Sampling range is required for fast sampling.'
            sampling_steps = list(sampling_range)
            z_0_list = self.fast_sampling(num_images, z_t, sampling_steps, condition=z_condition, show_verbose=show_verbose)
        else:
            z_0_list = DDPMManager.sampling(self, num_images, z_t, condition=z_condition, show_verbose=show_verbose)

        # exit latent space
        z_0_list = [img.unsqueeze(0) for img in z_0_list]
        z_0 = torch.cat(z_0_list, dim=0)
        x_0 = self.decode(z_0, *args)
        return [img for img in x_0]

    def train_step(self, x_train: torch.Tensor, y_train: torch.Tensor) -> dict[str, float]:
        # enter latent space
        z_x = self.encode(x_train)
        z_x = z_x if isinstance(z_x, torch.Tensor) else z_x[0]
        z_y = self.encode(y_train) if self.condition_encoder is None else self.condition_encoder(y_train)
        z_y = z_y if isinstance(z_y, torch.Tensor) else z_y[0]

        # forward diffusion model
        return DDPMManager.train_step(self, z_x, z_y)

    def test_step(self, x_test: torch.Tensor, y_test: torch.Tensor) -> dict[str, float]:
        # enter latent space
        z_x = self.encode(x_test)
        z_x = z_x if isinstance(z_x, torch.Tensor) else z_x[0]
        z_y = self.encode(y_test) if self.condition_encoder is None else self.condition_encoder(y_test)
        z_y = z_y if isinstance(z_y, torch.Tensor) else z_y[0]

        # forward diffusion model
        return DDPMManager.test_step(self, z_x, z_y)

    def to(self, device: torch.device) -> None:
        if self.condition_encoder is not None:
            self.condition_encoder = self.condition_encoder.to(device)
        return super().to(device)
