import torch


class Discriminator(torch.nn.Module):
    conv_out: torch.nn.Conv2d
    extractor: torch.nn.Sequential

    def __init__(self, in_channels: int, /, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()
        num_filters_mult = 1

        # extractor
        layers = [torch.nn.Conv2d(in_channels, num_filters_last, 4, 2, 1), torch.nn.LeakyReLU(0.2)]
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                torch.nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                torch.nn.BatchNorm2d(num_filters_last * num_filters_mult),
                torch.nn.LeakyReLU(0.2, True)
            ]
        self.extractor = torch.nn.Sequential(*layers)

        # output conv
        self.conv_out = torch.nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        return self.conv_out(x)
