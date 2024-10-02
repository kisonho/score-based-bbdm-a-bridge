import torch
import torch.nn.functional as F


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = torch.nn.Sequential(
            torch.nn.GroupNorm(32, in_channels),
            Swish(),
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            torch.nn.GroupNorm(32, out_channels),
            Swish(),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class Swish(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class UpSampleBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpSampleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super(DownSampleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = torch.nn.GroupNorm(32, channels)
        self.q = torch.nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = torch.nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = torch.nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A
