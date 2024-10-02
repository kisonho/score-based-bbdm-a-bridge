import torch

from .blocks import DownSampleBlock, NonLocalBlock, ResidualBlock


class Encoder(torch.nn.Module):
    down_sampling: torch.nn.Sequential
    mid_blocks: torch.nn.Sequential
    conv_out: torch.nn.Conv2d

    def __init__(self, in_channels: int, latent_dim: int = 3, f: int = 16):
        """
        Constructor

        - Parameters:
            - in_channels: An `int` of input image channels
            - latent_dim: An `int` of output latent dim
        """
        super(Encoder, self).__init__()
        channels = [128, 128, 256, 512]
        attn_resolutions = []
        num_res_blocks = 2
        resolution = 256

        # down sampling
        down_layers: list[torch.nn.Module] = [torch.nn.Conv2d(in_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for _ in range(num_res_blocks):
                down_layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    down_layers.append(NonLocalBlock(in_channels))
            
            # downsampling
            if i < torch.log2(torch.tensor(f)).numpy():
                down_layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        self.down_sampling = torch.nn.Sequential(*down_layers)

        # middle
        mid_blocks: list[torch.nn.Module] = []
        mid_blocks.append(ResidualBlock(channels[-1], channels[-1]))
        mid_blocks.append(NonLocalBlock(channels[-1]))
        mid_blocks.append(ResidualBlock(channels[-1], channels[-1]))
        mid_blocks.append(torch.nn.GroupNorm(32, channels[-1]))
        self.mid_blocks = torch.nn.Sequential(*mid_blocks)

        # out
        self.conv_out = torch.nn.Conv2d(channels[-1], latent_dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_sampling(x)
        x = self.mid_blocks(x)
        x *= x.sigmoid()
        return self.conv_out(x)