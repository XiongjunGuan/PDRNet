'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 19:42:09
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 20:18:45

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import torch


def show_pairs(img1, img2, mode="gray"):
    img1 = img1[:, :, None]
    img2 = img2[:, :, None]
    if mode == "rgb":
        if np.max(img1) <= 1:
            img0 = np.ones_like(img1)
        else:
            img0 = np.ones_like(img1) * 255.0
        img = np.concatenate((img0, img1, img2), axis=2)
    elif mode == "gray":
        img = (img1 * 1.0 + img2 * 1.0) / 2
    return img


def load_model(model, ckp_path):

    def remove_module_string(k):
        items = k.split(".")
        items = items[0:1] + items[2:]
        return ".".join(items)

    if isinstance(ckp_path, str):
        ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
        ckp_model_dict = ckp["model"]
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {
            remove_module_string(k): v
            for k, v in ckp_model_dict.items()
        }

    if hasattr(model, "module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)


def get_context_feature(magnitude, enh, use_intensity):
    if use_intensity is True:
        context_feature = torch.cat((enh, magnitude), dim=1).detach()
    else:
        context_feature = torch.cat((enh, ), dim=1).detach()
    return context_feature


def get_corr_feature(ori1, phase1, magnitude1, ori2, phase2, magnitude2,
                     use_intensity, use_orientation):
    if use_intensity is True:
        if use_orientation is True:
            corr_feature = torch.cat(
                (phase1 - phase2, ori1, ori2, magnitude1 * magnitude2),
                dim=1).detach()
        else:
            corr_feature = torch.cat(
                (phase1 - phase2, magnitude1 * magnitude2), dim=1).detach()
    else:
        if use_orientation is True:
            corr_feature = torch.cat((phase1 - phase2, ori1, ori2),
                                     dim=1).detach()
        else:
            corr_feature = torch.cat((phase1 - phase2, ), dim=1).detach()
    return corr_feature


def run_net(net, context1, context2, corr_feature, use_branch):
    if use_branch == "both":
        prob = net(
            context1,
            context2,
            corr_feature,
        )
    elif use_branch == "context":
        prob = net(
            context1,
            context2,
        )
    elif use_branch == "corr":
        prob = net(corr_feature, )
    return prob
