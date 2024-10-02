from torchmanager.losses import Loss
from torchmanager_core import torch
from torchmanager_core.typing import Optional, Union

from vqgan.data import VQData


class VQ(Loss):
    """
    The combined VQ loss

    - Parameters:
        - commitment_loss: The commitment loss that calculates loss between `sg[z_q]` and `z` in `Loss`
        - discrimination_loss: The GAN discrmination `Loss`
        - last_layer_weight: A `torch.Tensor` of the weight for the last decoder layer
        - main_loss: The main `Loss`
        - quantization_loss: The one that calculates loss between `sg[z]` and `z_q` in `Loss`
    """
    commitment_loss: Optional[Loss]
    gan_loss: Optional[Loss]
    last_layer_weight: torch.Tensor
    main_loss: Loss
    quantization_loss: Optional[Loss]

    def __init__(self, main_loss: Loss, quantization_loss: Optional[Loss] = None, commitment_loss: Optional[Loss] = None, gan_loss: Optional[Loss] = None, *, target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(target=target, weight=weight)
        self.commitment_loss = commitment_loss
        self.gan_loss = gan_loss
        self.main_loss = main_loss
        self.quantization_loss = quantization_loss

    def __call__(self, input: Union[VQData, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        return super().__call__(input, target)

    def forward(self, input: Union[VQData, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            return self.main_loss(input, target)
        else:
            # unpack data
            x, f, d = input

            # calculate perceptual loss
            loss = self.main_loss(x, target)

            # calculate gan loss
            if self.gan_loss is not None:
                loss += self.gan_loss(d.fake, loss)

            # calculate quantization loss
            if self.quantization_loss is not None:
                loss += self.quantization_loss(f.z_q, f.z.detach())

            # calculate commitment loss
            if self.commitment_loss is not None:
                loss += self.commitment_loss(f.z, f.z_q.detach())
            return loss

    def reset(self) -> None:
        # reset commitment loss
        if self.commitment_loss is not None:
            self.commitment_loss.reset()

        # reset gan loss
        if self.gan_loss is not None:
            self.gan_loss.reset()

        # reset main loss
        self.main_loss.reset()

        # reset quantization loss
        if self.quantization_loss is not None:
            self.quantization_loss.reset()
        return super().reset()
