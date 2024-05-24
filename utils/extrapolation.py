'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 19:42:09
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 20:17:27

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
from glob import glob

import numpy as np
import scipy.linalg
from scipy.ndimage import zoom

from utils.tps import tps_apply_transform, tps_module_numpy


def extrapolation_by_roi(dx, dy, roi):
    h, w = roi.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    zoom_param = 1 / 8
    dx_resize = zoom(dx, zoom=zoom_param, order=1)
    dy_resize = zoom(dy, zoom=zoom_param, order=1)
    mask_resize = zoom(roi, zoom=zoom_param, order=1)
    mask_resize = (mask_resize > 0.5)
    x_resize = zoom(x, zoom=zoom_param, order=1)
    y_resize = zoom(y, zoom=zoom_param, order=1)

    xs = x_resize[mask_resize > 0]
    ys = y_resize[mask_resize > 0]
    dxs = dx_resize[mask_resize > 0]
    dys = dy_resize[mask_resize > 0]

    src_cpts = np.float32(np.vstack((xs, ys)).T)
    src_pts = np.float32(
        np.vstack((x_resize.reshape((-1, )), y_resize.reshape((-1, )))).T)
    tar_cpts = np.float32(np.vstack(((xs + dxs, ys + dys))).T)

    mapping_matrix = tps_module_numpy(src_cpts, tar_cpts, 5)
    tar_pts = tps_apply_transform(src_pts, src_cpts, mapping_matrix)

    fx1, fx2 = x.shape[0] / x_resize.shape[0], x.shape[1] / x_resize.shape[1]
    fy1, fy2 = y.shape[0] / y_resize.shape[0], y.shape[1] / y_resize.shape[1]
    dx = zoom(tar_pts[:, 0].reshape(x_resize.shape), zoom=(fx1, fx2),
              order=1) - x
    dy = zoom(tar_pts[:, 1].reshape(y_resize.shape), zoom=(fy1, fy2),
              order=1) - y

    return dx, dy
