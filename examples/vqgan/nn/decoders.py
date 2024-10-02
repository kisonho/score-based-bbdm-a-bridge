import torch

from diffusion.nn import attention

from .blocks import NonLocalBlock, ResidualBlock, Swish, UpSampleBlock


class _Decoder(torch.nn.Module):
    conv_out: torch.nn.Conv2d
    norm: torch.nn.GroupNorm
    upsampling: torch.nn.Sequential

    def __init__(self, out_channels: int, latent_dim: int = 256, f: int = 16):
        super().__init__()
        channels = [512, 512, 256, 128]
        attn_resolutions = []
        num_res_blocks = 2
        resolution = 16

        # upsampling
        in_channels = channels[0]
        up_layers = [torch.nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            c = channels[i]
            for _ in range(num_res_blocks):
                up_layers.append(ResidualBlock(in_channels, c))
                in_channels = c
                if resolution in attn_resolutions:
                    up_layers.append(NonLocalBlock(in_channels))
            if i > len(channels) - torch.log2(torch.tensor(f)).numpy() - 1:
                up_layers.append(UpSampleBlock(in_channels))
                resolution *= 2
        self.upsampling = torch.nn.Sequential(*up_layers)

        # out
        self.norm = torch.nn.GroupNorm(32, in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsampling(x)
        x = self.norm(x)
        return self.conv_out(x)


class Decoder(torch.nn.Module):
    def __init__(self, out_channels: int, /, latent_dim: int = 3, f: int = 16):
        super().__init__()
        ch_mult = [1, 2, 4]
        ch = 128
        attn_resolutions = []
        self.num_resolutions = f-1
        self.num_res_blocks = 2
        resolution = 256

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,latent_dim,curr_res,curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(latent_dim, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = torch.nn.ModuleList()
        self.mid.append(ResidualBlock(in_channels=block_in, out_channels=block_in))
        self.mid.append(NonLocalBlock(block_in))
        self.mid.append(ResidualBlock(in_channels=block_in, out_channels=block_in))

        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks+1):
                block.append(ResidualBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(NonLocalBlock(block_in))
            up = torch.nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSampleBlock(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = torch.nn.GroupNorm(32, block_in)
        self.nonlinearity = Swish()
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        [block_1, attn_1, block_2] = self.mid
        h = block_1(h)
        h = attn_1(h)
        h = block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)  # type: ignore
                if len(self.up[i_level].attn) > 0:  # type: ignore
                    h = self.up[i_level].attn[i_block](h)  # type: ignore
            if i_level != 0:
                h = self.up[i_level].upsample(h)  # type: ignore

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h