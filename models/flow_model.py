"""
Description: 
Author: Xiongjun Guan
Date: 2023-04-20 19:35:46
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-05-11 22:11:24

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata

from models.units import DAPPM, ConvBnPRelu, ResidualBlock, ResidualBottleneck


class ContextEncoder(nn.Module):

    def __init__(self, in_chn):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBnPRelu(in_chn, 32, kernel_size=7, stride=2, padding=3))
        self.layer2 = ResidualBlock(32, 32, stride=1)
        self.layer3 = ResidualBlock(32, 48, stride=2)
        self.layer4 = ResidualBlock(48, 64, stride=2)
        self.layer5 = ResidualBlock(64, 96, stride=2)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class CorrEncoder(nn.Module):

    def __init__(self, in_chn):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBnPRelu(in_chn, 64, kernel_size=7, stride=2, padding=3))
        self.layer2 = ResidualBlock(64, 64, stride=1)
        self.layer3 = ResidualBlock(64, 96, stride=2)
        self.layer4 = ResidualBlock(96, 128, stride=2)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class FusionLayer_4(nn.Module):

    def __init__(self, in_chn, in_chn_):
        super().__init__()
        chn1 = in_chn * 1
        chn1_ = in_chn_ * 1
        self.layer1 = nn.Sequential(
            ResidualBlock(in_chn, chn1, stride=1),
            ResidualBlock(chn1, chn1),
        )
        self.layer1_ = nn.Sequential(
            ResidualBlock(in_chn_, chn1_),
            ResidualBlock(chn1_, chn1_),
        )
        self.from1 = nn.Sequential(
            nn.Conv2d(chn1, chn1_, kernel_size=1),
            nn.BatchNorm2d(chn1_),
        )
        self.from1_ = nn.Sequential(
            nn.Conv2d(chn1_, chn1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(chn1),
        )

        chn2 = round(in_chn * 1)
        chn2_ = round(in_chn_ * 1)
        self.layer2 = nn.Sequential(
            ResidualBlock(chn1, chn2),
            ResidualBlock(chn2, chn2),
        )
        self.layer2_ = nn.Sequential(
            ResidualBlock(chn1_, chn2_),
            ResidualBlock(chn2_, chn2_),
        )
        self.from2 = nn.Sequential(
            nn.Conv2d(chn2, chn2_, kernel_size=1),
            nn.BatchNorm2d(chn2_),
        )
        self.from2_ = nn.Sequential(
            nn.Conv2d(chn2_, chn2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(chn2),
        )

        chn3 = round(in_chn * 1)
        chn3_ = round(in_chn_ * 1)
        self.layer3 = nn.Sequential(
            ResidualBlock(chn2, chn3),
            ResidualBlock(chn3, chn3),
        )
        self.layer3_ = nn.Sequential(
            ResidualBlock(chn2_, chn3_),
            ResidualBlock(chn3_, chn3_),
        )
        self.from3 = nn.Sequential(
            nn.Conv2d(chn3, chn3_, kernel_size=1),
            nn.BatchNorm2d(chn3_),
        )
        self.from3_ = nn.Sequential(
            nn.Conv2d(chn3_, chn3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(chn3),
        )

        chn4 = round(in_chn * 1)
        chn4_ = round(in_chn_ * 1)
        self.layer4 = nn.Sequential(
            ResidualBlock(chn3, chn4),
            ResidualBlock(chn4, chn4),
        )
        self.layer4_ = nn.Sequential(
            ResidualBlock(chn3_, chn4_),
            ResidualBlock(chn4_, chn4_),
        )
        self.from4 = nn.Sequential(
            nn.Conv2d(chn4, chn4_, kernel_size=1),
            nn.BatchNorm2d(chn4_),
        )
        self.from4_ = nn.Sequential(
            nn.Conv2d(chn4_, chn4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(chn4),
        )

        chn5 = round(in_chn * 2)
        chn5_ = round(in_chn_ * 2)
        self.layer5 = nn.Sequential(ResidualBottleneck(chn4, chn4, chn5))
        self.layer5_ = nn.Sequential(ResidualBottleneck(chn4_, chn4_, chn5_))
        chn5 = round(in_chn * 2)

        self.spp = DAPPM(chn5, chn5_ // 2, chn5_)

        self.merge = nn.Sequential(ConvBnPRelu(chn5_, 256))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_):
        b, c, h, w = x_.shape
        layers = []
        x = self.layer1(x)
        layers.append(x)
        x_ = self.layer1_(x_)
        x = x + self.from1_(x_)
        x_ = x_ + F.interpolate(
            self.from1(layers[0]), size=(h, w), mode="bilinear")

        x = self.layer2(x)
        layers.append(x)
        x_ = self.layer2_(x_)
        x = x + self.from2_(x_)
        x_ = x_ + F.interpolate(
            self.from2(layers[1]), size=(h, w), mode="bilinear")

        x = self.layer3(x)
        layers.append(x)
        x_ = self.layer3_(x_)
        x = x + self.from3_(x_)
        x_ = x_ + F.interpolate(
            self.from3(layers[2]), size=(h, w), mode="bilinear")

        x = self.layer4(x)
        layers.append(x)
        x_ = self.layer4_(x_)
        x = x + self.from4_(x_)
        x_ = x_ + F.interpolate(
            self.from4(layers[3]), size=(h, w), mode="bilinear")

        x = self.spp(self.layer5(x))
        x = F.interpolate(x, size=(h, w), mode="bilinear")
        x_ = self.layer5_(x_)

        f = self.merge(x + x_)
        return f


class FlowHead_cla(nn.Module):

    def __init__(self, num_class=25):
        super().__init__()
        self.final_layer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, num_class * 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        probs = self.final_layer(x)
        ch = probs.shape[1] // 2
        proby = torch.softmax(probs[:, :ch, :, :], dim=1)
        probx = torch.softmax(probs[:, ch:, :, :], dim=1)
        probs = torch.cat((proby, probx), dim=1)
        return probs
