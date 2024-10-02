from typing import Union
import torch

from vqgan.data import DiscrminatedData, FeaturesData, VQData
from vqgan.nn import Discriminator, Decoder, Encoder, VectorQuantizer


class VQGAN(torch.nn.Module):
    """
    The VQGAN model
    Paper URL: https://arxiv.org/abs/2012.09841

    * extends: `torch.nn.Module`

    - Parameters:
        - decoder: A `vqgan.nn.Decoder` of the decoder
        - discriminator: A `.nn.Discriminator` of the discriminator
        - encoder: A `vqgan.nn.Encoder` of the encoder
        - post_quant_conv: A `torch.nn.Conv2d` of the quantized decoder convolutional layer
        - quant_conv: A `torch.nn.Conv2d` of the non-quantized encoder convolutional layer
        - quantizer: A `.nn.VectorQuantizer` of the quantization codebook
    """
    decoder: Decoder
    discriminator: Discriminator
    encoder: Encoder
    post_quant_conv: torch.nn.Conv2d
    quant_conv: torch.nn.Conv2d
    quantizer: VectorQuantizer
    return_details: bool

    def __init__(self, in_channels: int, *, f: int = 16, latent_dim: int = 3, num_codebook_vectors: int = 1024, return_details: bool = True) -> None:
        """
        Constructor

        - Parameters:
            - in_channels: An `int` of the input image dimension
            - latent_dim: An `int` of the latent domain dimension
            - num_codebook_vectors: An `int` of the number of codebook vectors
            - return_details: A `bool` flag of if returning detail data in wrap of `data.VQData`
        """
        super().__init__()
        self.return_details = return_details

        # initialize encoder
        self.encoder = Encoder(in_channels, latent_dim=latent_dim, f=f)

        # initialize decoder
        self.decoder = Decoder(in_channels, latent_dim=latent_dim, f=f)

        # initialize quantizer, including a quantization convolutional layer, vector quantizer, and a post-quantization convolutional layer
        self.quantizer = VectorQuantizer(num_codebook_vectors, latent_dim=latent_dim)
        self.quant_conv = torch.nn.Conv2d(latent_dim, latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_dim, latent_dim, 1)

        # initialize discriminator
        self.discriminator = Discriminator(in_channels)

    def __call__(self, x: torch.Tensor) -> Union[VQData, torch.Tensor]:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> Union[VQData, torch.Tensor]:
        # forwarding, enc -> quant_conv -> quantizer -> post_quant_conv -> dec
        z: torch.Tensor = self.encoder(x)
        z = self.quant_conv(z)
        z_q: torch.Tensor = self.quantizer(z)
        z_q = self.post_quant_conv(z_q)
        x_hat: torch.Tensor = self.decoder(z_q)

        # pass to discriminator
        d_fake = self.discriminator(x_hat)
        d_real = self.discriminator(x)

        # wrap data
        f = FeaturesData(z, z_q)
        d = DiscrminatedData(d_fake, d_real)
        y = VQData(x_hat, f, d)
        return y if self.return_details else x_hat
