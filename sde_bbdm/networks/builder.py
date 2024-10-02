from typing import Sequence

from .openai import UNet


def build_unet(in_channels: int, out_channels: int, /, *, attention_resolutions: Sequence[int] = [32, 16, 8], channel_mults: Sequence[int] = [1, 4, 8]) -> UNet:
    return UNet(in_channels, 128, out_channels, num_res_blocks=2, attention_resolutions=attention_resolutions, channel_mult=channel_mults, num_heads=8, num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True)
