'''
Description: 
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 20:47:20

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import os
import os.path as osp
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def show_pairs(bimg1, bimg2):
    h, w = bimg1.shape
    img = np.ones((h, w, 3)) * 255
    b1 = (bimg1 < 128)
    b2 = (bimg2 < 128)
    common = b1 & b2
    only1 = b1 & (~b2)
    only2 = (~b1) & b2

    img[:, :, 0][common] = 0
    img[:, :, 1][~common] = 255
    img[:, :, 2][common] = 0

    img[:, :, 0][only1] = 128
    img[:, :, 1][only1] = 128
    img[:, :, 2][only1] = 128

    img[:, :, 0][only2] = 0
    img[:, :, 1][only2] = 0
    img[:, :, 2][only2] = 255

    return img


if __name__ == "__main__":
    ftitle1_lst = ["1a", "2a"]
    ftitle2_lst = ["1b", "2b"]
    save_show_dir = "./examples/results"
    if not osp.exists(save_show_dir):
        os.makedirs(save_show_dir)

    bimg_dir = "./examples/data_bimg/"
    regist_bimg_dir = "./examples/results_bimg/"

    for ftitle1, ftitle2 in zip(ftitle1_lst, ftitle2_lst):
        bimg1 = cv2.imread(osp.join(bimg_dir, ftitle1 + ".png"), 0)
        bimg2 = cv2.imread(osp.join(bimg_dir, ftitle2 + ".png"), 0)
        show_img = show_pairs(bimg1, bimg2)
        cv2.imwrite(osp.join(save_show_dir, f"{ftitle1}-{ftitle2}_tps.png"),
                    show_img)

        bimg2 = cv2.imread(osp.join(regist_bimg_dir, ftitle2 + ".png"), 0)
        show_img = show_pairs(bimg1, bimg2)
        cv2.imwrite(osp.join(save_show_dir, f"{ftitle1}-{ftitle2}_PDRNet.png"),
                    show_img)
