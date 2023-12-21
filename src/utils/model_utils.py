import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelShuffleUpsampling(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, scale_factor : int):
        super(PixelShuffleUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * scale_factor * scale_factor, kernel_size=3, padding=1)

    def forward(self, inputs):
        out_feature = self.conv(inputs)
        return F.pixel_shuffle(out_feature, self.scale_factor)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DWSConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DWSConvolution, self).__init__()
        self.module = nn.Sequential(
            DepthwiseConvolution(in_channels, kernel_size, stride, padding),
            PointwiseConvolution(in_channels, out_channels, stride)
        )

    def forward(self, inputs):
        return self.module(inputs)

class DepthwiseConvolution(nn.Module):
    def __init__(self, channels : int, kernel_size : int = 3, stride : int = 1, padding : int = 1):
        super(DepthwiseConvolution, self).__init__()
        self.module = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=channels)

    def forward(self, inputs):
        return self.module(inputs)

class PointwiseConvolution(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, stride : int = 1):
        super(PointwiseConvolution, self).__init__()
        self.module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                padding=0)

    def forward(self, inputs):
        return self.module(inputs)

def GetIrradiance(color : torch.Tensor, BRDF : torch.Tensor) -> torch.Tensor:
    return torch.where(BRDF == 0, BRDF, color / BRDF)

def backward_warping(data : torch.Tensor, motion : torch.Tensor) -> torch.Tensor:
    return F.grid_sample(data, get_grid(motion), mode='nearest', align_corners=False, padding_mode = "zeros")

def get_grid(motion : torch.Tensor)-> torch.Tensor:
    _, _, height, width = motion.size()

    hori = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(-1, -1, height, -1)
    verti = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(-1, -1, -1, width)
    grid = torch.cat([hori, verti], 1).cuda()


    vgrid = grid + motion
    vgrid = vgrid.permute((0, 2, 3, 1))

    return vgrid

def check_nan(t : torch.tensor)->bool:
    return not torch.any(torch.isnan(t)).item()

def check_inf(t : torch.tensor)->bool:
    return not torch.any(torch.isinf(t)).item()

def check_infNnan(t : torch.tensor)->bool:
    return check_nan(t) and check_inf(t)