'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 19:42:10
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 20:48:34

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import copy
import os
import os.path as osp
import random
import warnings
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.models import Cla2Reg, PDRNet_L4, RidgeNet, SpatialTransformer
from utils.extrapolation import extrapolation_by_roi
from utils.utils import get_corr_feature, load_model, run_net

warnings.filterwarnings("ignore")


def set_seed(seed=7):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed(7)
    cuda_ids = [2]
    batch_size = 1

    dst_shape = (512, 512)  # image size

    # set model path
    config_path = "./ckpts/config.yaml"  # PDRNet
    model_path = "./ckpts/PDRNet_L4.pth"  # PDRNet
    enh_model_path = "./ckpts/ridge.pth"  # EnhNet

    # set data path
    # Note that the fingerprint pair has been roughly registered using TPS based on matching minutiae
    img1_dirs = ["./examples/data_img/"]
    img2_dirs = ["./examples/data_img/"]
    bimg1_dirs = ["./examples/data_bimg/"]
    bimg2_dirs = ["./examples/data_bimg/"]
    mask1_dirs = ["./examples/data_mask/"]
    mask2_dirs = ["./examples/data_mask/"]
    ftitle1_lst = ["2a"]
    ftitle2_lst = ["2b"]

    # set save dir
    save_img_dir = "./examples/results_img/"
    save_bimg_dir = "./examples/results_bimg/"
    save_mask_dir = "./examples/results_mask/"
    if not osp.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not osp.exists(save_bimg_dir):
        os.makedirs(save_bimg_dir)
    if not osp.exists(save_mask_dir):
        os.makedirs(save_mask_dir)

    device = torch.device("cuda:{}".format(str(cuda_ids[0])) if torch.cuda.
                          is_available() else "cpu")

    # load PDRNet model config
    with open(config_path, "r") as config:
        cfg = yaml.safe_load(config)
    use_intensity = cfg["model_cfg"]["use_intensity"]
    use_orientation = cfg["model_cfg"]["use_orientation"]
    use_branch = cfg["model_cfg"]["use_branch"]
    step = cfg["model_cfg"]["step"]
    num_class = cfg["model_cfg"]["num_class"]

    context_chn = 1
    corr_chn = 1
    if cfg["model_cfg"]["use_intensity"] is True:
        corr_chn += 1
    if cfg["model_cfg"]["use_orientation"] is True:
        corr_chn += 2

    if cfg["model_cfg"]["model"] == "PDRNet_L4":
        if cfg["model_cfg"]["use_branch"] == "both":
            model = PDRNet_L4(
                num_class=cfg["model_cfg"]["num_class"],
                context_chn=context_chn,
                corr_chn=corr_chn,
            )

    model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model,
                                  device_ids=cuda_ids,
                                  output_device=cuda_ids[0])
    model = model.to(device)
    model.eval()

    # load EnhNet model config
    enh_model = RidgeNet()
    load_model(enh_model, enh_model_path)
    enh_model = enh_model.to(device)
    enh_model.eval()

    # functions
    trans_func = SpatialTransformer(size=dst_shape)
    trans_Cla2Reg = Cla2Reg(step=step)

    with torch.no_grad():
        pbar = tqdm(range(len(ftitle1_lst)))
        for i in pbar:
            try:
                img1_dir = img1_dirs[i]
                img2_dir = img2_dirs[i]

                bimg1_dir = bimg1_dirs[i]
                bimg2_dir = bimg2_dirs[i]

                mask1_dir = mask1_dirs[i]
                mask2_dir = mask2_dirs[i]

                ftitle1 = ftitle1_lst[i]
                ftitle2 = ftitle2_lst[i]

                # read data
                img1 = cv2.imread(osp.join(img1_dir, ftitle1 + ".png"),
                                  0).astype(np.float32)
                img2 = cv2.imread(osp.join(img2_dir, ftitle2 + ".png"),
                                  0).astype(np.float32)
                mask1 = cv2.imread(osp.join(mask1_dir, ftitle1 + ".png"),
                                   0).astype(np.float32)
                mask2 = cv2.imread(osp.join(mask2_dir, ftitle2 + ".png"),
                                   0).astype(np.float32)
                bimg1 = cv2.imread(osp.join(bimg1_dir, ftitle1 + ".png"),
                                   0).astype(np.float32)
                bimg2 = cv2.imread(osp.join(bimg2_dir, ftitle2 + ".png"),
                                   0).astype(np.float32)

                border_mask = np.ones_like(img2)
                border_mask[2:-2, 2:-2] = 0
                img2[border_mask == 1] = 255
                bimg2[border_mask == 1] = 255
                mask2[border_mask == 1] = 0

                # preprocess
                img1 = (255 - img1) / 255.0
                img2 = (255 - img2) / 255.0
                bimg1 = (255 - bimg1) / 255.0
                bimg2 = (255 - bimg2) / 255.0
                ori_bimg2 = copy.deepcopy(bimg2)

                mask1 = (mask1 > 0.5)
                mask2 = (mask2 > 0.5)
                mask = mask1 * mask2
                bimg1 *= mask
                bimg2 *= mask

                bimg1 = torch.from_numpy(bimg1[None,
                                               None, :, :]).float().to(device)
                bimg2 = torch.from_numpy(bimg2[None,
                                               None, :, :]).float().to(device)
                img2 = torch.from_numpy(img2[None,
                                             None, :, :]).float().to(device)
                ori_bimg2 = torch.from_numpy(
                    ori_bimg2[None, None, :, :]).float().to(device)
                mask = torch.from_numpy(mask[None,
                                             None, :, :]).float().to(device)
                mask2 = torch.from_numpy(mask2[None,
                                               None, :, :]).float().to(device)

                # orientation, phase, magnitude, enhanced fingprint
                [ori1, ori2], [phase1,
                               phase2], [magnitude1, magnitude2
                                         ], [enh1,
                                             enh2] = enh_model([bimg1, bimg2])

                # correlation information
                corr_feature = get_corr_feature(ori1, phase1, magnitude1, ori2,
                                                phase2, magnitude2,
                                                use_intensity, use_orientation)

                # prob_pred: [b, 2*num_class, h/8, w/8]
                # where 2*num_class represents the magnitude of displacement in the form of interval classification
                prob_pred = run_net(model, enh1, enh2, corr_feature,
                                    use_branch)

                # up_prob: [b, 2*num_class, h, w]
                up_prob = F.interpolate(prob_pred,
                                        size=dst_shape,
                                        mode="bilinear")

                # flow_pred: [b, 2, h, w]
                # convert classification to numerical value
                flow_pred = trans_Cla2Reg(up_prob)

                # expanding non overlapping regions through interpolation
                flow_pred = flow_pred.detach()
                dy = flow_pred[:, 0, :, :]
                dx = flow_pred[:, 1, :, :]
                dx = dx.cpu().numpy().squeeze()
                dy = dy.cpu().numpy().squeeze()

                dx, dy = extrapolation_by_roi(dx, dy,
                                              mask.cpu().numpy().squeeze())
                dx = torch.from_numpy(dx[None, None, :, :]).float().to(device)
                dy = torch.from_numpy(dy[None, None, :, :]).float().to(device)

                # flow_pred: [b, 2, h, w]
                flow_pred = torch.cat((dy, dx), dim=1)

                # transform img
                img2_regist = trans_func(img2, flow_pred)
                bimg2_regist = trans_func(ori_bimg2, flow_pred)
                mask2_regist = trans_func(mask2, flow_pred)

                # save results
                img2_regist = (
                    1 - img2_regist).detach().squeeze().cpu().numpy() * 255
                bimg2_regist = (
                    1 - bimg2_regist).detach().squeeze().cpu().numpy() * 255
                mask2_regist = (mask2_regist).detach().squeeze().cpu().numpy()
                mask2_regist = (mask2_regist > 0.5)

                img2_regist = np.clip(img2_regist, 0, 255)
                bimg2_regist = np.clip(bimg2_regist, 0, 255)
                mask2_regist = np.clip(mask2_regist * 255, 0, 255)

                cv2.imwrite(osp.join(save_img_dir, f"{ftitle2}.png"),
                            img2_regist)
                cv2.imwrite(osp.join(save_bimg_dir, f"{ftitle2}.png"),
                            bimg2_regist)
                cv2.imwrite(osp.join(save_mask_dir, f"{ftitle2}.png"),
                            mask2_regist)

            except:
                continue

        pbar.close()
