'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 19:42:09
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 20:18:35

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import itertools
from glob import glob

import cv2
import numpy as np
import scipy
import scipy.linalg
import torch
import torch.nn as nn
from scipy.interpolate import RectBivariateSpline
from torch.autograd import Variable


# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    pairwise_diff = input_points[:, None] - control_points[None]
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[..., 0] + pairwise_diff_square[..., 1]
    # fix numerical error for 0 * log(0), substitute all nan with 0
    repr_matrix = 0.5 * pairwise_dist * np.log(pairwise_dist.clip(1e-3, None))
    mask = (repr_matrix != repr_matrix) | np.isclose(pairwise_dist, 0)
    repr_matrix[mask] = 0
    return repr_matrix


def opencv_tps(img, source, target, mode=1, border_value=0):
    tps = cv2.createThinPlateSplineShapeTransformer()
    # tps.setRegularizationParameter(0.01)

    source_ts = source[None]
    target_ts = target[None]
    matches = [cv2.DMatch(ii, ii, 0) for ii in range(len(source))]
    tps.estimateTransformation(target_ts, source_ts, matches)
    if mode == 0:
        flags = cv2.INTER_NEAREST
    elif mode == 1:
        flags = cv2.INTER_LINEAR

    return tps.warpImage(img, flags=flags, borderValue=border_value)


def normalization(points, img_shape):
    points = points * 1.0 / (np.array(img_shape)[None, ::-1] - 1) * 2 - 1
    return points


def tps_module_numpy(src_cpts, tar_cpts, Lambda=0):
    assert tar_cpts.ndim == 2
    assert tar_cpts.shape[1] == 2
    N = src_cpts.shape[0]
    src_cpts = src_cpts.astype(np.float32)
    tar_cpts = tar_cpts.astype(np.float32)

    # create padded kernel matrix
    src_cc_partial_repr = compute_partial_repr(src_cpts,
                                               src_cpts) + Lambda * np.eye(N)
    forward_kernel = np.concatenate(
        (
            np.concatenate(
                (src_cc_partial_repr, np.ones([N, 1]), src_cpts), axis=1),
            np.concatenate((np.ones([1, N]), np.zeros([1, 3])), axis=1),
            np.concatenate((src_cpts.T, np.zeros([2, 3])), axis=1),
        ),
        axis=0,
    )
    # compute mapping matrix
    Y = np.concatenate([tar_cpts, np.zeros([3, 2])], axis=0)  # (M+3,2)
    mapping_matrix = scipy.linalg.solve(forward_kernel, Y)
    return mapping_matrix


def tps_apply_transform(src_pts, src_cpts, mapping_matrix):
    """
    Parameters:
        src_pts: points to be transformed
        src_cpts: control points
    Returns:
        [None]
    """
    assert src_pts.ndim == 2
    src_pc_partial_repr = compute_partial_repr(src_pts, src_cpts)
    N = src_pts.shape[0]
    src_pts_repr = np.concatenate(
        [src_pc_partial_repr, np.ones([N, 1]), src_pts], axis=1)
    tar_pts = np.matmul(src_pts_repr, mapping_matrix)
    return tar_pts


def fast_tps_transform(
    img,
    tar_shape,
    flow,
    matches,
    center=None,
    theta=0,
    shift=np.zeros(2),
    rotation=0,
    stride=16,
    interpolation=cv2.INTER_LINEAR,
):
    tar_center = np.array([tar_shape[1], tar_shape[0]]) / 2
    if center is None:
        center = np.array([img.shape[1], img.shape[0]]) / 2
    R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    R_rotation = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation),
                            np.cos(rotation)]])

    src_x, src_y = np.meshgrid(np.linspace(-200, 200, 11),
                               np.linspace(-160, 160, 9))
    src_x = src_x.T.reshape(-1)
    src_y = src_y.T.reshape(-1)
    # tar_x = src_x + flow[0::2]
    # tar_y = src_y + flow[1::2]
    src_cpts = np.stack((src_x, src_y), axis=-1)
    tar_cpts = src_cpts + flow.reshape(-1, 2)

    src_cpts = src_cpts.dot(R_theta.T) + center[None]
    tar_cpts = tar_cpts.dot(R_theta.T) + tar_center[None]

    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(tar_cpts[None], src_cpts[None], matches)
    grid_x = np.arange(-stride, tar_shape[1] + stride * 2 - 2, step=stride)
    grid_y = np.arange(-stride, tar_shape[0] + stride * 2 - 2, step=stride)
    tar_pts = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).astype(np.float32)
    src_pts = tps.applyTransformation(tar_pts.reshape(
        1, -1, 2))[1].reshape(*tar_pts.shape)

    bspline_x = RectBivariateSpline(grid_y, grid_x, src_pts[..., 0])
    bspline_y = RectBivariateSpline(grid_y, grid_x, src_pts[..., 1])
    tps_x, tps_y = np.meshgrid(np.arange(tar_shape[1]),
                               np.arange(tar_shape[0]))
    tps_x = bspline_x.ev(tps_y, tps_x).astype(np.float32)
    tps_y = bspline_y.ev(tps_y, tps_x).astype(np.float32)

    tps_pts = (np.stack((tps_x, tps_y), axis=-1) - center[None] -
               shift[None]).dot(R_rotation) + center[None]
    tps_pts = tps_pts.astype(np.float32)
    img_tps = cv2.remap(img, tps_pts[..., 0], tps_pts[..., 1], interpolation)
    return img_tps


def compute_partial_repr_for_torch(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :,
                                                                         1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


def trans_pts_from_np_to_torch(pts, image_height, image_width):
    """ (0,h) -> (-1,1)
        (0,w) -> (-1,1)

    Args:
        pts (_type_): _description_
        h (_type_): _description_
        w (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert len(pts.shape) == 2
    assert pts.shape[1] == 2
    pts[:, 0] = (pts[:, 0] - image_height / 2) / (image_height / 2)
    pts[:, 1] = (pts[:, 1] - image_width / 2) / (image_width / 2)

    return torch.from_numpy(pts)


class TPSGridGen(nn.Module):
    """_summary_
    src_cpts = trans_pts_from_np_to_torch(src_cpts,h,w)
    tar_cpts = trans_pts_from_np_to_torch(tar_cpts,h,w)
    tar_cpts = torch.unsqueeze(tar_cpts,dim=0)

    tps = TPSGridGen(h,w,src_cpts)
    tar_pts = tps(tar_cpts)
    
    img = torch.tensor(np.float32(img/255))[np.newaxis,np.newaxis,:,:]
    batch_size = tar_pts.size(0)
    coordinate = tar_pts.view(batch_size, image_height, image_width, 2)
    img = F.grid_sample(img, coordinate,align_corners=False)
    img = torch.squeeze(img,dim=1).cpu().numpy()[0,:,:]
    img = np.uint8(img*255)

    Args:
        nn (_type_): _description_
    """

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr_for_torch(
            target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(
            itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y],
                                      dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr_for_torch(
            target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr,
            torch.ones(HW, 1), target_coordinate
        ],
                                           dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([
            source_control_points,
            Variable(self.padding_matrix.expand(batch_size, 3, 2))
        ], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr),
                                         mapping_matrix)
        return source_coordinate


if __name__ == "__main__":
    prefix = ""
