from typing import Iterable
from torch.utils.data import DataLoader
from torchmanager.data import Dataset
from diffusion import DiffusionData
from diffusion.managers import DiffusionManager
from torch.nn.modules.module import Module
from torchmanager import losses, metrics
from torchmanager_core import abc, devices, torch, view
from torchmanager_core.typing import Any, Generic, Module, Optional, Sequence, TypeVar, Union, overload

E = TypeVar('E', bound=Optional[torch.nn.Module])
D = TypeVar('D', bound=Optional[torch.nn.Module])


class LatentDiffusionManager(DiffusionManager[Module], Generic[Module, E, D], abc.ABC):
    """
    Diffusion manager for diffusion models under latent space as an option.

    * extends: `DiffusionManager`
    * generic: `Module`, `E`, `D`
    * abstract class

    - Parameters:
        - encoder: An optional encoder in `E` to enter latent space.
        - decoder: An decoder in `D` to exit latent space.

    - Methods to implement:
        - Abstract methods are inherited from `DiffusionManager`.
    """
    encoder: Union[E, torch.nn.DataParallel[torch.nn.Module]]
    decoder: Union[D, torch.nn.DataParallel[torch.nn.Module]]

    def __init__(self, model: Module, time_steps: int, /, encoder: E = None, decoder: D = None, *, optimizer: Optional[torch.optim.Optimizer] = None, loss_fn: Optional[Union[losses.Loss, dict[str, losses.Loss]]] = None, metrics: dict[str, metrics.Metric] = {}) -> None:
        """
        Constructor

        - Prarameters:
            - model: An optional target `torch.nn.Module` to be trained
            - time_steps: An `int` of total number of steps
            - encoder: An optional encoder in `E` to enter latent space
            - decoder: An optional decoder in `D` to exit latent space
            - optimizer: An optional `torch.optim.Optimizer` to train the model
            - loss_fn: An optional `Loss` object to calculate the loss for single loss or a `dict` of losses in `Loss` with their names in `str` to calculate multiple losses
            - metrics: An optional `dict` of metrics with a name in `str` and a `Metric` object to calculate the metric
        """
        super().__init__(model, time_steps, optimizer, loss_fn, metrics)

        # initialize encoder
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder.eval()

        # initialize decoder
        self.decoder = decoder
        if self.decoder is not None:
            self.decoder.eval()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, *args: Any) -> torch.Tensor:
        if self.decoder is None:
            return z
        return self.decoder(z, *args)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        if self.encoder is None:
            return x
        return self.encoder(x)

    @torch.no_grad()
    def fast_sampling(self, num_images: int, x_t: torch.Tensor, sampling_range: Sequence[int], condition: Optional[torch.Tensor] = None, *, show_verbose: bool = False) -> list[torch.Tensor]:
        '''
        Samples a given number of images using fast sampling algorithm.

        - Parameters:
            - num_images: An `int` of number of images to generate
            - x_t: A `torch.Tensor` of the image at T step
            - sampling_range: An `Iterable[int]` of the range of time steps to sample
            - condition: An optional `torch.Tensor` of the condition to generate images
            - start_index: An optional `int` of the start index of reversed time step
            - end_index: An `int` of the end index of reversed time step
            - show_verbose: A `bool` flag to show the progress bar during testing
        - Retruns: A `list` of `torch.Tensor` generated results
        '''
        # initialize
        progress_bar = view.tqdm(desc='Sampling loop time step', total=len(sampling_range)) if show_verbose else None

        # sampling loop time step
        for i, tau in enumerate(sampling_range):
            # fetch data
            t = torch.full((num_images,), tau, dtype=torch.long, device=x_t.device)
            tau_minus_one = sampling_range[i+1] if i < len(sampling_range) - 1 else 0

            # append to predicitions
            x = DiffusionData(x_t, t, condition=condition)
            y = self.fast_sampling_step(x, tau, tau_minus_one)
            assert isinstance(y, torch.Tensor), "The output must be a valid `torch.Tensor`."
            x_t = y.to(x_t.device)

            # update progress bar
            if progress_bar is not None:
                progress_bar.update()

        # return final image
        x_0 = x_t
        return [img for img in x_0]
    
    @overload
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...

    @overload
    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = True, predicted_obj: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        '''
        Samples a single time step using fast sampling algorithm.

        - Parameters:
            - data: A `DiffusionData` of the data to sample
            - tau: An `int` of the current time step
            - tau_minus_one: An `int` of the next time step
            - return_noise: A `bool` flag to return noise
            - predicted_obj: An optional `torch.Tensor` of the predicted object
        - Returns: A `torch.Tensor` of the sampled image or a `tuple` of `torch.Tensor` of the sampled image and `torch.Tensor` of the noise
        '''
        raise NotImplementedError('Fast sampling step method has not been implemented yet.')

    def sampling(self, num_images: int, x_t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False) -> list[torch.Tensor]:
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
        z_t = self.encode(x_t)
        z_t, _ = (z_t, None) if isinstance(z_t, torch.Tensor) else z_t
        z_condition = self.encode(condition)
        z_condition, args = (z_condition, None) if isinstance(z_condition, torch.Tensor) else z_condition

        # sampling
        if fast_sampling:
            assert sampling_range is not None, 'Sampling range is required for fast sampling.'
            sampling_steps = list(sampling_range)
            z_0_list = self.fast_sampling(num_images, z_t, sampling_steps, condition=z_condition, show_verbose=show_verbose)
        else:
            assert not isinstance(sampling_range, list), 'Sampling range must be a `range` or `reversed` for original sampling.'
            z_0_list = super().sampling(num_images, z_t, condition=z_condition, sampling_range=sampling_range, show_verbose=show_verbose)

        # exit latent space
        z_0_list = [img.unsqueeze(0) for img in z_0_list]
        z_0 = torch.cat(z_0_list, dim=0)
        x_0 = self.decode(z_0, *args)
        return [img for img in x_0]

    def data_parallel(self, target_devices: list[torch.device]) -> bool:
        # data parallel encoder
        if self.decoder is not None:
            self.decoder, decoder_use_multi_gpus = devices.data_parallel(self.decoder, devices=target_devices)  # type: ignore
        else:
            decoder_use_multi_gpus = True

        # data parallel decoder
        if self.encoder is not None:
            self.encoder, encoder_use_multi_gpus = devices.data_parallel(self.encoder, devices=target_devices)  # type: ignore
        else:
            encoder_use_multi_gpus = True

        # update use_multi_gpus
        use_multi_gpus = decoder_use_multi_gpus and encoder_use_multi_gpus
        return super().data_parallel(target_devices) and use_multi_gpus

    @overload
    def predict(self, num_images: int, image_size: Union[int, tuple[int, ...]], *args: Any, condition: Optional[torch.Tensor] = None, noises: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Sequence[int], range]] = None, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        ...

    @overload
    def predict(self, num_images: int, image_size: Union[int, tuple[int, ...]], *args: Any, condition: Optional[torch.Tensor] = None, noises: Optional[torch.Tensor] = None, fast_sampling: bool = True, sampling_range: Sequence[int], device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = True, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        ...

    def predict(self, num_images: int, image_size: Union[int, tuple[int, ...]], *args: Any, condition: Optional[torch.Tensor] = None, noises: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Sequence[int], range]] = None, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> list[torch.Tensor]:
        return super().predict(num_images, image_size, *args, condition=condition, noises=noises, fast_sampling=fast_sampling, sampling_range=sampling_range, device=device, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, **kwargs)

    def reset(self, cpu: torch.device = devices.CPU) -> None:
        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder = self.encoder.module.to(cpu)  # type: ignore
        if isinstance(self.decoder, torch.nn.DataParallel):
            self.decoder = self.decoder.module.to(cpu)  # type: ignore
        return super().reset(cpu)

    def train_step(self, x_train: torch.Tensor, y_train: torch.Tensor) -> dict[str, float]:
        # enter latent space
        z_x = self.encode(x_train)
        z_x, _ = (z_x, None) if isinstance(z_x, torch.Tensor) else z_x
        z_y = self.encode(y_train)
        z_y, _ = (z_y, None) if isinstance(z_y, torch.Tensor) else z_y

        # forward diffusion model
        return super().train_step(z_x, z_y)

    @overload
    def test(self, dataset: Union[DataLoader[torch.Tensor], Dataset[torch.Tensor]], *args: Any, sampling_images: bool = False, fast_sampling: bool = False, sampling_shape: Optional[Union[int, tuple[int, ...]]] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, device: Optional[Union[torch.device, list[torch.device]]] = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        ...

    @overload
    def test(self, dataset: Union[DataLoader[torch.Tensor], Dataset[torch.Tensor]], *args: Any, sampling_images: bool = True, fast_sampling: bool = True, sampling_shape: Optional[Union[int, tuple[int, ...]]] = None, sampling_range: Sequence[int], device: Optional[Union[torch.device, list[torch.device]]] = None, empty_cache: bool = True, use_multi_gpus: bool = True, show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        ...

    def test(self, dataset: Union[DataLoader[torch.Tensor], Dataset[torch.Tensor]], *args: Any, sampling_images: bool = False, fast_sampling: bool = False, sampling_shape: Optional[Union[int, tuple[int, ...]]] = None, sampling_range: Optional[Union[Sequence[int], range]] = None, device: Optional[Union[torch.device, list[torch.device]]] = None, empty_cache: bool = True, use_multi_gpus: bool = False, show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        """
        Test target model

        - Parameters:
            - dataset: A `torch.utils.data.DataLoader` or `.data.Dataset` dataset
            - sampling_images: A `bool` flag to sample images during testing
            - *args: An optional `tuple` of `Any` of additional arguments for sampling
            - fast_sampling: A `bool` flag to use fast sampling during testing
            - sampling_shape: An optional `int` or `tuple` of `int` of the shape of sampled images
            - sampling_range: An optional `Iterable[int]`, `range`, or `reversed` of the range of time steps to sample
            - device: An optional `torch.device` to test on if not using multi-GPUs or an optional default `torch.device` for testing otherwise
            - empyt_cache: A `bool` flag to empty cache after testing
            - use_multi_gpus: A `bool` flag to use multi gpus during testing
            - show_verbose: A `bool` flag to show the progress bar during testing
            - **kwargs: An optional `dict` of `Any` of additional keyword arguments for sampling
        - Returns: A `dict` of validation summary
        """
        return super().test(dataset, *args, sampling_images=sampling_images, sampling_shape=sampling_shape, fast_sampling=fast_sampling, sampling_range=sampling_range, device=device, empty_cache=empty_cache, use_multi_gpus=use_multi_gpus, show_verbose=show_verbose, **kwargs)

    def test_step(self, x_test: torch.Tensor, y_test: torch.Tensor) -> dict[str, float]:
        # enter latent space
        z_x = self.encode(x_test)
        z_x, _ = (z_x, None) if isinstance(z_x, torch.Tensor) else z_x
        z_y = self.encode(y_test)
        z_y, _ = (z_y, None) if isinstance(z_y, torch.Tensor) else z_y

        # forward diffusion model
        return super().test_step(z_x, z_y)

    def to(self, device: torch.device) -> None:
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
        if self.decoder is not None:
            self.decoder = self.decoder.to(device)
        super().to(device)
