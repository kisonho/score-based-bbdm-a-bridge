import torch
from torchmanager.losses import Loss


class _ScalingLayer(torch.nn.Module):
    scale: torch.Tensor
    shift: torch.Tensor

    def __init__(self):
        super(_ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class _NetLinLayer(torch.nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(_NetLinLayer, self).__init__()
        layers = [torch.nn.Dropout(), ] if (use_dropout) else []
        layers += [torch.nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = torch.nn.Sequential(*layers)


class LPIPS(Loss):
    net: torch.nn.Module

    def __init__(self, net: torch.nn.Module, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = _ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = net
        self.lin0 = _NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = _NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = _NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = _NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = _NetLinLayer(self.chns[4], use_dropout=use_dropout)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for i in range(len(self.chns)):
            feats0[i], feats1[i] = _normalize_tensor(outs0[i]), _normalize_tensor(outs1[i])
            diffs[i] = feats0[i] - feats1[i]

        res = [_spatial_average(lins[i].model(diffs[i]) ** 2, keepdim=True) for i in range(len(self.chns))]
        val: torch.Tensor = res[0]
        for i in range(1, len(self.chns)):
            val += res[i]
        return val.sum()


def _normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def _spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

