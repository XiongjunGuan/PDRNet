'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 19:42:09
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 20:16:41

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


class NormalizeModule(nn.Module):

    def __init__(self, m0, var0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m)**2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class ConvBnPRelu(nn.Module):

    def __init__(self,
                 in_chn,
                 out_chn,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chn,
            out_chn,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_chn)
        self.relu = nn.PReLU(out_chn)

    def forward(self, input):
        y = self.conv(input)
        y = self.bn(y)
        y = self.relu(y)
        return y


class ResidualBlock(nn.Module):

    def __init__(self, in_chn, out_chn, stride=1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride,
                      padding=1),
            nn.BatchNorm2d(out_chn),
            nn.PReLU(out_chn),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
        )
        self.relu = nn.PReLU(out_chn)

        if (in_chn != out_chn) or (stride != 1):
            self.in_conv = nn.Sequential(
                nn.Conv2d(in_chn,
                          out_chn,
                          kernel_size=3,
                          stride=stride,
                          padding=1))
        else:
            self.in_conv = None

    def forward(self, x):
        if self.in_conv is None:
            identity = x
        else:
            identity = self.in_conv(x)

        out = self.layer1(x)
        out = self.layer2(out)
        out += identity
        out = self.relu(out)

        return out


class ResidualBottleneck(nn.Module):

    def __init__(self, in_chn, mid_chn, out_chn, stride=1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chn, mid_chn, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_chn),
            nn.PReLU(mid_chn),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_chn,
                      mid_chn,
                      kernel_size=3,
                      stride=stride,
                      padding=1),
            nn.BatchNorm2d(mid_chn),
            nn.PReLU(mid_chn),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_chn, out_chn, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_chn),
        )
        self.relu = nn.PReLU(out_chn)

        if (in_chn != out_chn) or (stride != 1):
            self.in_conv = nn.Sequential(
                nn.Conv2d(in_chn,
                          out_chn,
                          kernel_size=3,
                          stride=stride,
                          padding=1))
        else:
            self.in_conv = None

    def forward(self, x):
        if self.in_conv is None:
            identity = x
        else:
            identity = self.in_conv(x)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += identity
        out = self.relu(out)
        return out


class DAPPM(nn.Module):

    def __init__(self, in_chn, mid_chn, out_chn):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_chn),
            nn.PReLU(),
            nn.Conv2d(in_chn, mid_chn, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_chn),
            nn.PReLU(),
            nn.Conv2d(in_chn, mid_chn, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(in_chn),
            nn.PReLU(),
            nn.Conv2d(in_chn, mid_chn, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_chn),
            nn.PReLU(),
            nn.Conv2d(in_chn, mid_chn, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(mid_chn),
            nn.PReLU(),
            nn.Conv2d(mid_chn, mid_chn, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(mid_chn),
            nn.PReLU(),
            nn.Conv2d(mid_chn, mid_chn, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(mid_chn),
            nn.PReLU(),
            nn.Conv2d(mid_chn, mid_chn, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(mid_chn * 4),
            nn.PReLU(),
            nn.Conv2d(mid_chn * 4, out_chn, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_chn),
            nn.PReLU(),
            nn.Conv2d(in_chn, out_chn, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x), size=[height, width], mode="bilinear") +
                           x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x), size=[height, width], mode="bilinear") +
                                      x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x), size=[height, width], mode="bilinear") +
                           x_list[2])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


def gabor_bank(enh_ksize=25,
               ori_stride=2,
               sigma=4.5,
               Lambda=8,
               psi=0,
               gamma=0.5):
    grid_theta, grid_x, grid_y = torch.meshgrid(
        torch.arange(-90, 90, ori_stride, dtype=torch.float32),
        torch.arange(-(enh_ksize // 2),
                     enh_ksize // 2 + 1,
                     dtype=torch.float32),
        torch.arange(-(enh_ksize // 2),
                     enh_ksize // 2 + 1,
                     dtype=torch.float32),
        indexing="ij",
    )
    cos_theta = torch.cos(torch.deg2rad(-grid_theta))
    sin_theta = torch.sin(torch.deg2rad(-grid_theta))

    x_theta = grid_y * sin_theta + grid_x * cos_theta
    y_theta = grid_y * cos_theta - grid_x * sin_theta
    # gabor filters
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    exp_fn = torch.exp(-0.5 *
                       (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
    gb_cos = exp_fn * torch.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * torch.sin(2 * np.pi * x_theta / Lambda + psi)

    return gb_cos[:, None], gb_sin[:, None]


def cycle_gaussian_weight(ang_stride=2, to_tensor=True):
    gaussian_pdf = signal.windows.gaussian(181, 3)
    coord = np.arange(ang_stride / 2, 180, ang_stride)
    delta = np.abs(coord.reshape(1, -1, 1, 1) - coord.reshape(-1, 1, 1, 1))
    delta = np.minimum(delta, 180 - delta) + 90
    if to_tensor:
        return torch.tensor(gaussian_pdf[delta.astype(int)]).float()
    else:
        return gaussian_pdf[delta.astype(int)].astype(np.float32)


def orientation_highest_peak(x, ang_stride=2):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    filter_weight = cycle_gaussian_weight(ang_stride=ang_stride).type_as(x)
    return F.conv2d(x, filter_weight, stride=1, padding=0)


def select_max_orientation(x):
    x = x / torch.max(x, dim=1, keepdim=True).values.clamp_min(1e-8)
    x = torch.where(x > 0.999, x, torch.zeros_like(x))
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return x
