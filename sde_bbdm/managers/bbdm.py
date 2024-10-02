from typing import Iterable
import torch
from diffusion import DiffusionData
from torchmanager_core import torch
from torchmanager_core.typing import Module, Optional, Union

from .latent import LatentDiffusionManager, E, D


class BBDMManager(LatentDiffusionManager[Module, E, D]):
    """Diffusion manager for the BBDM."""

    def forward_diffusion(self, data: torch.Tensor, condition: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None) -> tuple[
        DiffusionData, torch.Tensor]:
        # step1 create t
        x_start = data
        batch_size = x_start.shape[0]
        t = torch.randint(1, self.time_steps, (batch_size,), device=data.device).long() if t is None else t.long()

        # step2 create noise
        noise = torch.randn_like(x_start, device=x_start.device)
        assert condition is not None, 'Condition is required for forward diffusion.'

        # calculate x_t
        m_t = t / self.time_steps
        m_t = m_t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # BCWH
        delta_t = 2 * (m_t - m_t ** 2)
        xt = (1 - m_t) * x_start + m_t * condition + delta_t ** 0.5 * noise
        objective = m_t * (condition - x_start) + delta_t ** 0.5 * noise
        return DiffusionData(xt, t), objective

    def sampling(self, num_images: int, x_t: torch.Tensor, /, *, condition: Optional[torch.Tensor] = None, fast_sampling: bool = False, sampling_range: Optional[Union[Iterable[int], reversed, range]] = None, show_verbose: bool = False) -> list[torch.Tensor]:
        x_t = condition if condition is not None else x_t
        sampling_range = reversed(range(1, self.time_steps)) if sampling_range is None else sampling_range
        return super().sampling(num_images, x_t, condition=condition, fast_sampling=fast_sampling, sampling_range=sampling_range, show_verbose=show_verbose)

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False, predicted_noise: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # m_t = t/T
        t = i
        m_t = t / self.time_steps
        m_t = torch.full((data.x.shape[0],), m_t, device=data.x.device)
        m_t = m_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # m_t_minus_one
        m_t_minus_one = (t-1) / self.time_steps
        m_t_minus_one = torch.full((data.x.shape[0],), m_t_minus_one, device=data.x.device)
        m_t_minus_one = m_t_minus_one.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # delta_t and delta_t_minus_one
        delta_t = 2 * (m_t - m_t ** 2)
        delta_t_minus_one = 2 * (m_t_minus_one - m_t_minus_one ** 2)
        delta_t_by_t_minus_one = delta_t - delta_t_minus_one * ((1 - m_t) ** 2) / ((1 - m_t_minus_one) ** 2)
        tilde_delta_t = delta_t_by_t_minus_one * delta_t_minus_one / delta_t

        # c_xt, c_yt, and c_epst
        c_xt = (delta_t_minus_one / delta_t) * (1 - m_t) / (1 - m_t_minus_one) + delta_t_by_t_minus_one / delta_t * (1 - m_t_minus_one)
        c_yt = m_t_minus_one - m_t * (1 - m_t) / (1 - m_t_minus_one) * (delta_t_minus_one / delta_t)
        c_epst = (1 - m_t_minus_one) * delta_t_by_t_minus_one / delta_t

        if predicted_noise is None: 
            predicted_noise, _ = self.forward(data)
            assert predicted_noise is not None, "Predicted noise must be given."
        # predict noise

        # initialize new noise
        new_noise = torch.randn_like(data.x, device=data.x.device) if t > 1 else 0

        # sampling equation
        assert data.condition is not None, "Condition must be given."
        x_t_minus_one = c_xt * data.x + c_yt * data.condition - c_epst * predicted_noise + tilde_delta_t ** 0.5 * new_noise
        # x_t_minus_one = c_xt * data.x + c_yt * data.condition - c_epst * predicted_noise
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
            sigma2_t = (var_tau - var_tau_minus_one * (1. - m_tau) ** 2 / (1. - m_tau_minus_one) ** 2) * var_tau_minus_one / var_tau
            sigma_t = torch.sqrt(sigma2_t)

            # calculate x_t_minus_one
            noise = torch.randn_like(data.x)
            x_tminus_mean = (1. - m_tau_minus_one) * x0_recon + m_tau_minus_one * data.condition + torch.sqrt((var_tau_minus_one - sigma2_t) / var_tau) * (data.x - (1. - m_tau) * x0_recon - m_tau * data.condition)
            x_t_minus_one = x_tminus_mean + sigma_t * noise
        return (x_t_minus_one, predicted_obj) if return_noise else x_t_minus_one
