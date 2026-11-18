# Derived from the supplied project. Ultralytics AGPL-3.0; see LICENSE.
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ultralytics.nn.modules.block import Attention, C2f, C3, Bottleneck, C3k, PSABlock
from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .mamba import GatedCNNBlock_BCHW

class UnderwaterEnhanceC3K2(C2f):
    """C3k2 with lightweight underwater enhancement (no IO change)."""

    def __init__(self, c1: int, c2: int, n: int=1, c3k: bool=False, e: float=0.5, attn: bool=False, g: int=1, shortcut: bool=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden_c = self.c
        self.m = nn.ModuleList((nn.Sequential(Bottleneck(hidden_c, hidden_c, shortcut, g), UnderwaterPSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))) if attn else C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)))

class UnderwaterPSABlock(nn.Module):
    """PSA block with lightweight underwater feature calibration."""

    def __init__(self, c: int, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        x = x * self.scale + self.bias
        return x

class ECA(nn.Module):
    """
    Efficient Channel Attention
    YOLOv26 compatible
    """

    def __init__(self, c1, c2=None, gamma=2, b=1, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        k = int(abs((math.log(c1, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class SimAM(nn.Module):
    """
    Simple Attention Module
    """

    def __init__(self, c1=None, c2=None, e_lambda=0.0001, *args, **kwargs):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = max(h * w - 1, 1)
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)

class ShuffleAttention(nn.Module):
    """
    Shuffle Attention
    """

    def __init__(self, c1, c2=None, g=8, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        assert c1 % (2 * g) == 0, f'channels must be divisible by 2*g, got c1={c1}, g={g}'
        self.g = g
        self.c = c1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, c1 // (2 * g), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, c1 // (2 * g), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, c1 // (2 * g), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, c1 // (2 * g), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.g, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        xs = self.sweight * x_1 + self.sbias
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        out = out.view(b, self.g, c // self.g, h, w)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, c, h, w)
        return out

class UnderwaterEnhance(nn.Module):
    """
    Underwater Enhancement Module
    """

    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        self.wb_conv = nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
        hidden = max(c2 // 2, 8)
        self.contrast_conv = nn.Sequential(nn.Conv2d(c2, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(), nn.Conv2d(hidden, c2, 3, padding=1, bias=False), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x):
        wb = self.wb_conv(x)
        gamma_corrected = torch.pow(torch.clamp(wb.float(), min=0.0001), self.gamma.float()).to(wb.dtype)
        contrast = self.contrast_conv(gamma_corrected)
        enhanced = gamma_corrected * contrast
        out = gamma_corrected + 0.3 * enhanced
        return out

class MultiScaleFusion(nn.Module):
    """
    Multi-scale Feature Fusion
    """

    def __init__(self, c1, c2=None, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        self.cv1 = Conv(c1, c2, 1)
        self.cv3 = Conv(c1, c2, 3)
        self.cv5 = Conv(c1, c2, 5)
        self.fuse = Conv(c2 * 3, c2, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv3(x)
        x3 = self.cv5(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.fuse(out)

class GhostConv_Underwater(nn.Module):
    """
    Underwater GhostConv
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, *args, **kwargs):
        super().__init__()
        c_ = math.ceil(c2 / 2)
        self.cv1 = Conv(c1, c_, k, s)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_)
        self.cv3 = Conv(c_ * 2, c2, 1)
        self.att = SimAM()

    def forward(self, x):
        y = self.cv1(x)
        y2 = self.cv2(y)
        out = torch.cat([y, y2], dim=1)
        out = self.cv3(out)
        return self.att(out)

class DySnakeConv(nn.Module):
    """
    Lightweight Dynamic Snake Conv
    """

    def __init__(self, c1, c2, k=3, s=1, *args, **kwargs):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, stride=s, padding=k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.att = ECA(c2)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.att(x)

class C2f_Underwater(nn.Module):
    """
    Underwater C2f
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, use_enhance=True, use_att=True, *args, **kwargs):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList((Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
        self.use_enhance = use_enhance
        self.use_att = use_att
        if use_enhance:
            self.enhance = UnderwaterEnhance(self.c)
        if use_att:
            self.att = ECA(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        if self.use_enhance:
            y[0] = self.enhance(y[0])
            y[1] = self.enhance(y[1])
        y.extend((m(y[-1]) for m in self.m))
        out = self.cv2(torch.cat(y, 1))
        if self.use_att:
            out = self.att(out)
        return out

class MambaBlock(nn.Module):
    """
    Lightweight Vision Mamba Block
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.proj_x = nn.Conv2d(dim, dim, 1)
        self.proj_z = nn.Conv2d(dim, dim, 1)
        self.dwconv_f = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dwconv_b = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x_proj = self.proj_x(x)
        z = self.proj_z(x)
        y_f = self.dwconv_f(x_proj)
        y_b = torch.flip(x_proj, dims=[-1])
        y_b = self.dwconv_b(y_b)
        y_b = torch.flip(y_b, dims=[-1])
        gate = self.act(z)
        y = (y_f + y_b) * gate
        y = self.out_proj(y)
        return y + identity

class C2f_MambaOut(C2f):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList((GatedCNNBlock_BCHW(self.c) for _ in range(n)))

class DeformConv(nn.Module):

    def __init__(self, in_channels, kernel_size=(3, 3), padding=1, stride=1, dilation=1, groups=None, bias=False):
        super().__init__()
        if groups is None:
            groups = in_channels
        self.offset_net = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=True)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class deformable_LKA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.ones(1) * 0.01)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = torch.sigmoid(attn)
        return shortcut + self.gamma * (shortcut * attn)

class Bottleneck_DLKA(Bottleneck):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = deformable_LKA(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3_DLKA(C3):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DLKA(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DLKA(C2f):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList((Bottleneck_DLKA(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)))
CUSTOM_LAYERS = {name: globals()[name] for name in ['UnderwaterEnhanceC3K2', 'UnderwaterPSABlock', 'ECA', 'SimAM', 'ShuffleAttention', 'UnderwaterEnhance', 'MultiScaleFusion', 'GhostConv_Underwater', 'DySnakeConv', 'C2f_Underwater', 'MambaBlock', 'C2f_MambaOut', 'DeformConv', 'deformable_LKA', 'Bottleneck_DLKA', 'C3_DLKA', 'C2f_DLKA']}
