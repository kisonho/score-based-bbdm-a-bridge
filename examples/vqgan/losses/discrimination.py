from torch.nn import functional as F
from torchmanager.losses import Loss
from torchmanager_core import torch
from torchmanager_core.typing import Optional, Union

from vqgan.data import VQData


class Discrimination(Loss):
    """The discrimination loss"""
    def __init__(self, *, target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(target=target, weight=weight)

    def __call__(self, input: Union[VQData, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the discrimination loss
        
        - Parameters:
            - input: A `tuple` of VQGAN output
            - target: A `torch.Tensor
        """
        if isinstance(input, VQData):
            return super().__call__(input.d.fake, input.d.real)
        else:
            return super().__call__(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d_loss_fake = torch.mean(F.relu(1 + input))
        d_loss_real = torch.mean(F.relu(1 - target))
        return 0.5 * (d_loss_real + d_loss_fake)


class GAN(Loss):
    """
    The GAN loss

    - Properties:
        - last_layer_weight: A `torch.Tensor` of the weight for the last layer of decoder
    """
    last_layer_weight: torch.Tensor

    def __init__(self, w, /, *, target: Optional[str] = None, weight: float = 1) -> None:
        """
        Constructor

        - Parameters:
            - last_layer_weight: A `torch.Tensor` of the weight for the last layer of decoder
            - target: An optional `str` of target name in `input` and `target` during direct calling
            - weight: A `float` of the loss weight
        """
        super().__init__(target=target, weight=weight)
        self.last_layer_weight = w

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the GAN loss
        
        - Parameters:
            - input: The fake discriminated logits
            - target: The perceptual loss
        """
        return super().__call__(input, target)

    def calculate_lambda(self, perceptual_loss: torch.Tensor, gan_loss: torch.Tensor) -> torch.Tensor:
        """
        - Calculate the adaptive weights (lambda)

        - Parameters:
            - perceptual_loss: A `torch.Tensor` of the perceptual loss
            - gan_loss: A `torch.Tensor` of the gan loss
        """
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, self.last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, self.last_layer_weight, retain_graph=True)[0]
        adaptive_weight = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        adaptive_weight = torch.clamp(adaptive_weight, 0, 1e4).detach()
        return 0.8 * adaptive_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = - input.mean()
        w = self.calculate_lambda(target, loss) if self.training else 1
        return w * loss
