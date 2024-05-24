"""
Description: 
Author: Xiongjun Guan
Date: 2023-04-13 17:12:44
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-04-19 21:01:00

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata

from models.flow_model import (ContextEncoder, CorrEncoder, FlowHead_cla,
                               FusionLayer_4)
from models.units import (ConvBnPRelu, NormalizeModule, cycle_gaussian_weight,
                          gabor_bank, orientation_highest_peak,
                          select_max_orientation)


class PDRNet_L4(nn.Module):
    """Phase-aggregated Dual-branch Registration Network

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_class=25, context_chn=1, corr_chn=3):
        super().__init__()
        self.context_layer = ContextEncoder(in_chn=context_chn)
        self.corr_layer = CorrEncoder(in_chn=corr_chn)
        self.fusion_layer = FusionLayer_4(in_chn=192, in_chn_=128)
        self.flow_layer = FlowHead_cla(num_class=num_class)

    def forward(self, img1, img2, corr_feature):
        [context_feature1, context_feature2] = self.context_layer([img1, img2])
        corr_feature = self.corr_layer(corr_feature)
        context_feature = torch.cat([context_feature1, context_feature2],
                                    axis=1)
        x = self.fusion_layer(context_feature, corr_feature)
        prob = self.flow_layer(x)

        return prob


class RidgeNet(nn.Module):
    """Part of FingerNet

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.img_norm = NormalizeModule(m0=0, var0=1)

        # feature extraction VGG
        self.conv1 = nn.Sequential(ConvBnPRelu(1, 64, 3),
                                   ConvBnPRelu(64, 64, 3), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(ConvBnPRelu(64, 128, 3),
                                   ConvBnPRelu(128, 128, 3),
                                   nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            ConvBnPRelu(128, 256, 3),
            ConvBnPRelu(256, 256, 3),
            ConvBnPRelu(256, 256, 3),
            nn.MaxPool2d(2, 2),
        )

        # multi-scale ASPP
        self.conv4_1 = ConvBnPRelu(256, 256, 3, padding=1, dilation=1)
        self.ori1 = nn.Sequential(
            ConvBnPRelu(256, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 90, 1, stride=1, padding=0),
        )

        self.conv4_2 = ConvBnPRelu(256, 256, 3, padding=4, dilation=4)
        self.ori2 = nn.Sequential(
            ConvBnPRelu(256, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 90, 1, stride=1, padding=0),
        )

        self.conv4_3 = ConvBnPRelu(256, 256, 3, padding=8, dilation=8)
        self.ori3 = nn.Sequential(
            ConvBnPRelu(256, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 90, 1, stride=1, padding=0),
        )

        # enhance part
        gabor_cos, gabor_sin = gabor_bank(enh_ksize=25, ori_stride=2, Lambda=8)

        self.enh_img_real = nn.Conv2d(gabor_cos.size(1),
                                      gabor_cos.size(0),
                                      kernel_size=(25, 25),
                                      padding=12)
        self.enh_img_real.weight = nn.Parameter(gabor_cos, requires_grad=True)
        self.enh_img_real.bias = nn.Parameter(torch.zeros(gabor_cos.size(0)),
                                              requires_grad=True)

        self.enh_img_imag = nn.Conv2d(gabor_sin.size(1),
                                      gabor_sin.size(0),
                                      kernel_size=(25, 25),
                                      padding=12)
        self.enh_img_imag.weight = nn.Parameter(gabor_sin, requires_grad=True)
        self.enh_img_imag.bias = nn.Parameter(torch.zeros(gabor_sin.size(0)),
                                              requires_grad=True)

    def forward(self, input):
        is_list = isinstance(input, tuple) or isinstance(input, list)
        if is_list:
            batch_dim = input[0].shape[0]
            input = torch.cat(input, dim=0)

        img_norm = self.img_norm(input)

        # feature extraction VGG
        conv1 = self.conv1(img_norm)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # multi-scale ASPP
        conv4_1 = self.conv4_1(conv3)
        ori1 = self.ori1(conv4_1)

        conv4_2 = self.conv4_2(conv3)
        ori2 = self.ori2(conv4_2)

        conv4_3 = self.conv4_3(conv3)
        ori3 = self.ori3(conv4_3)

        ori_out = torch.sigmoid(ori1 + ori2 + ori3)

        # enhance part
        enh_real = self.enh_img_real(input)
        enh_imag = self.enh_img_imag(input)
        ori_peak = orientation_highest_peak(ori_out)
        ori_peak = select_max_orientation(ori_peak)

        ori_up = F.interpolate(ori_peak,
                               size=(enh_real.shape[2], enh_real.shape[3]),
                               mode="nearest")
        enh_real = (enh_real * ori_up).sum(1, keepdim=True)
        enh_imag = (enh_imag * ori_up).sum(1, keepdim=True)

        Z = torch.complex(enh_real, enh_imag)
        phase = torch.angle(Z)
        magnitude = torch.abs(Z)
        enh = torch.cos(phase)

        ori = (torch.argmax(ori_up, dim=1, keepdim=True) * 2.0 -
               90) / 90  # set [-90, 90] to [-1, 1]
        magnitude /= 24  # set blank area to 1.0

        if is_list:
            ori = torch.split(ori, [batch_dim, batch_dim], dim=0)
            phase = torch.split(phase, [batch_dim, batch_dim], dim=0)
            magnitude = torch.split(magnitude, [batch_dim, batch_dim], dim=0)
            enh = torch.split(enh, [batch_dim, batch_dim], dim=0)

        return ori, phase, magnitude, enh


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        self.grid = torch.stack(grids)
        self.grid = torch.unsqueeze(self.grid, 0)
        self.grid = self.grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid.to(flow.device) + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i,
                     ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Reg2Cla(nn.Module):

    def __init__(self, step=2.5, sigma=1.5, eps=1e-6):
        super().__init__()
        self.step = step
        self.sigma = sigma
        self.eps = eps

    def forward(self, gt, num_class=25):
        threshod = ((num_class - 1) // 2) * self.step
        dy_gt = gt[:, 0:1, :, :]
        dx_gt = gt[:, 1:2, :, :]
        grid_d = torch.linspace(-threshod, threshod, num_class)[None, :, None,
                                                                None]
        grid_d = grid_d.repeat(1, 1, dx_gt.shape[2],
                               dx_gt.shape[3]).to(dx_gt.device)
        dx_tar = grid_d - dx_gt
        dy_tar = grid_d - dy_gt
        dx_tar = torch.exp(-(dx_tar**2) / (2 * self.sigma**2))
        dy_tar = torch.exp(-(dy_tar**2) / (2 * self.sigma**2))
        dx_tar = dx_tar / dx_tar.sum(dim=1, keepdim=True).clamp_min(self.eps)
        dy_tar = dy_tar / dy_tar.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return dy_tar, dx_tar


class Cla2Reg(nn.Module):

    def __init__(self, step=2.5):
        super().__init__()
        self.step = step

    def forward(self, prob):
        b, c, h, w = prob.shape
        num_classes = c // 2
        threshod = ((num_classes - 1) // 2) * self.step

        dy_prob = prob[:, :num_classes, :, :]
        dx_prob = prob[:, num_classes:, :, :]

        grid_d = torch.linspace(-threshod, threshod, num_classes)[None, :,
                                                                  None, None]
        grid_d = grid_d.repeat(1, 1, dx_prob.shape[2],
                               dx_prob.shape[3]).to(prob.device)
        dy = torch.sum(dy_prob * grid_d, dim=1, keepdim=True)
        dx = torch.sum(dx_prob * grid_d, dim=1, keepdim=True)
        reg = torch.cat((dy, dx), dim=1)

        return reg
