import torch
from torch.optim.optimizer import Optimizer as Optimizer
from torchmanager.losses.loss import Loss
from torchmanager.metrics.metric import Metric
from diffusion import DiffusionData
from torchmanager_core import torch
from torchmanager_core.typing import Module, Optional, Sequence, Union
import math

from .bbdm import BBDMManager
from .latent import LatentDiffusionManager, E, D


class SDEBBDMManager(LatentDiffusionManager[Module, E, D]):
    """Diffusion manager for the A-Bridge BBDM with algorithm in sec. 1 and 2 offered by Prof. Wang."""

    c_lambda: float

    def __init__(self, model: Module, time_steps: int, /, encoder: E = None, decoder: D = None, c_lambda: float = 2, *, optimizer: Optional[Optimizer] = None, loss_fn: Optional[Union[Loss, dict[str, Loss]]] = None, metrics: dict[str, Metric] = {}) -> None:
        super().__init__(model, time_steps, encoder, decoder, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)
        self.c_lambda = c_lambda

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[
        DiffusionData, torch.Tensor]:
        # step1 create t
        x_start = data
        batch_size = x_start.shape[0]
        T = self.time_steps
        t = torch.randint(2, self.time_steps + 1, (batch_size,), device=data.device).long() if t is None else t.long()
        t_reshaped = t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # BCWH
        m_t = t_reshaped / self.time_steps
        # print(t_reshaped)
                # step2 create noise
        noise = torch.randn_like(x_start, device=x_start.device)
        assert condition is not None, 'Condition is required for forward diffusion.'
        B_t = self.c_lambda * (1 - m_t) * (torch.log(1 / (1 - m_t))) ** 0.5
        B_t = torch.where(torch.eq(t_reshaped, T), torch.zeros_like(B_t), B_t)
        # print(B_t)
        xt = (1 - m_t) * x_start + m_t * condition +  B_t * noise
        objective = m_t * (condition - x_start) + B_t * noise
        return DiffusionData(xt, t), objective

    def sampling(self, num_images: int, x_t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False) -> list[torch.Tensor]:
        # check if condition has been given
        if condition is not None:
            x_t = condition
        else:
            condition = x_t

        # sampling
        return super().sampling(num_images, x_t, condition=condition, fast_sampling=fast_sampling, sampling_range=sampling_range, show_verbose=show_verbose)

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # m_t = t/T
        t = i
        T = self.time_steps
        m_t = t / T
        m_t = torch.full((data.x.shape[0],), m_t, device=data.x.device)
        m_t = m_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # predict noise
        if predicted_obj is None: 
            predicted_obj, _ = self.forward(data)
            assert predicted_obj is not None, "Predicted noise must be given."

        # initialize new noise
        new_noise = torch.randn_like(data.x, device=data.x.device) if t > 2 else 0

        # sampling equation
        assert data.condition is not None, "Condition must be given."
        if i == 1:
            x_t_minus_one: torch.Tensor = data.x - predicted_obj
        else:
            beta_t = T - t + 1
            gamma_t = math.log(T / beta_t)
            C = 1 - 1 / beta_t + 1 / (beta_t * gamma_t)
            c_xt = 1 / C * (1 + 1 / (T * gamma_t))
            c_yt = 1 / C * (1 / beta_t - (t - 1) / (T * beta_t * gamma_t))
            c_epst = 1 / C * (1 / (T * gamma_t))
            c_zt = self.c_lambda / C * (1 - m_t + 1 / T) ** 0.5 * (1 / T) ** 0.5
            x_t_minus_one: torch.Tensor = c_xt * data.x - c_yt * data.condition - c_epst * predicted_obj - c_zt * new_noise

        return (x_t_minus_one, predicted_obj) if return_noise else x_t_minus_one   

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # predict noise
        if predicted_obj is None: 
            predicted_obj, _ = self.forward(data)
            assert predicted_obj is not None, "Predicted noise must be given."

        # unpack data
        assert data.condition is not None, "Condition must be given."
        x_tau = data.x
        y = data.condition

        # if this is the last step
        if tau_minus_one == 0:
            x_tau_minus_one: torch.Tensor = x_tau - predicted_obj
            return (x_tau_minus_one, predicted_obj) if return_noise else x_tau_minus_one

        # m_t = t/T
        T = self.time_steps
        m_tau = tau / T
        m_tau = torch.full((x_tau.shape[0],), m_tau, device=x_tau.device)
        m_tau = m_tau.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # m_{t-1} = (t-1)/T
        m_tau_minus_one = tau_minus_one / T
        m_tau_minus_one = torch.full((data.x.shape[0],), m_tau_minus_one, device=data.x.device)
        m_tau_minus_one = m_tau_minus_one.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # y_{t-1} and y_t
        y_tau_minus_one = (1 - m_tau_minus_one) * (x_tau - predicted_obj) + m_tau_minus_one * y
        y_tau = (1 - m_tau) * (x_tau - predicted_obj) + m_tau * y

        # initialize new noise
        new_noise = torch.randn_like(x_tau, device=x_tau.device)

        # sampling equation
        B_tau_minus_one = self.c_lambda * (1 - m_tau_minus_one) * (torch.log(1 / (1 - m_tau_minus_one))) ** 0.5
        B_tau = self.c_lambda * (1 - m_tau) * (torch.log(1 / (1 - m_tau))) ** 0.5 if tau != T else 1
        sigma_tau = 0.5 * B_tau_minus_one
        x_tau_minus_one = y_tau_minus_one + (torch.sqrt(B_tau_minus_one ** 2 - sigma_tau ** 2) / B_tau) * (x_tau - y_tau) + sigma_tau * new_noise
        return (x_tau_minus_one, predicted_obj) if return_noise else x_tau_minus_one  


class BBDMSpecialCaseManager(BBDMManager[Module, E, D]):
    """Diffusion manager for the BBDM."""

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[DiffusionData, torch.Tensor]:
        t = torch.randint(1, self.time_steps + 1, (data.shape[0],), device=data.device).long() if t is None else t.long()
        return super().forward_diffusion(data, condition, t)
    
    def sampling(self, num_images: int, x_t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Sequence[int], range]] = None, show_verbose: bool = False) -> list[torch.Tensor]:
        sampling_range = range(self.time_steps, 0, -1) if sampling_range is None else sampling_range
        return super().sampling(num_images, x_t, condition=condition, fast_sampling=fast_sampling, sampling_range=sampling_range, show_verbose=show_verbose)

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False, predicted_noise: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # m_t = t/T
        t = i
        m_t = t / self.time_steps
        m_t = torch.full((data.x.shape[0],), m_t, device=data.x.device)
        m_t = m_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # c_xt, c_yt, and c_epst, tilde_delta_t
        c_xt = 1
        c_epst = 1 / t
        c_noise = (2 / self.time_steps) ** 0.5

        # check if predicted noise has already been given
        if predicted_noise is None: 
            predicted_noise, _ = self.forward(data)
            assert predicted_noise is not None, "Predicted noise must be given."

        # initialize new noise
        new_noise = torch.randn_like(data.x, device=data.x.device) if t > 1 else 0

        # sampling equation
        assert data.condition is not None, "Condition must be given."
        x_t_minus_one = c_xt * data.x - c_epst * predicted_noise - c_noise * new_noise
        return (x_t_minus_one, predicted_noise) if return_noise else x_t_minus_one

    def fast_sampling_step(self, data: DiffusionData, tau: int, tau_minus_one: int, /, *, return_noise: bool = False, predicted_obj: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # predict noise
        objective_recon, _ = self.forward(data)
        predicted_obj = objective_recon if predicted_obj is None else predicted_obj
        assert predicted_obj is not None, "Predicted noise must be given."

        # fast sampling
        if tau == 1:
            x_t_minus_one = data.x - predicted_obj
        else:
            # reconstruct x0
            x0_recon = data.x - predicted_obj

            # m_t
            m_tau = tau / self.time_steps
            m_tau = torch.full((data.x.shape[0],), m_tau, device=data.x.device)
            m_tau = m_tau.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            # m_t_minus_one
            m_tau_minus_one = tau_minus_one / self.time_steps
            m_tau_minus_one = torch.full((data.x.shape[0],), m_tau_minus_one, device=data.x.device)
            m_tau_minus_one = m_tau_minus_one.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            # var_tau and var_tau_minus_one
            var_tau = 2. * (m_tau - m_tau ** 2)
            var_tau_minus_one = 2. * (m_tau_minus_one - m_tau_minus_one ** 2)

            # sigma^2_t and sigma_t
            sigma_t = 0.5 * var_tau_minus_one

            # calculate x_t_minus_one
            noise = torch.randn_like(data.x)
            x_tminus_mean = (1. - m_tau_minus_one) * x0_recon + m_tau_minus_one * data.condition + torch.sqrt((var_tau_minus_one - sigma_t ** 2) / var_tau) * (data.x - (1. - m_tau) * x0_recon - m_tau * data.condition)
            x_t_minus_one = x_tminus_mean + sigma_t * noise
        return (x_t_minus_one, predicted_obj) if return_noise else x_t_minus_one


BBDMSDELinerRefinedWithLambdaV2With1000 = SDEBBDMManager
