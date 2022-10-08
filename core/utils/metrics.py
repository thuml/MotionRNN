__author__ = 'yunbo'

import numpy as np
import copy


def cal_csi(pd, gt, level):
    # [w,h]
    pdf = pd.astype(np.float32)
    gtf = gt.astype(np.float32)
    pd_ = np.zeros(pd.shape)
    gt_ = np.zeros(gt.shape)
    pd_[(pdf + 30) / 2 >= level] = 1
    gt_[(gtf + 30) / 2 >= level] = 1
    csi_ = pd_ + gt_
    if (csi_ >= 1).sum() == 0:
        return 0.0
    return float((csi_ == 2).sum()) / float((csi_ >= 1).sum())


def cal_far(pd, gt, level):
    # [w,h]
    pdf = pd.astype(np.float32)
    gtf = gt.astype(np.float32)
    pd_ = np.zeros(pd.shape)
    gt_ = np.zeros(gt.shape)
    pd_[(pdf + 30) / 2 >= level] = 1
    gt_[(gtf + 30) / 2 >= level] = 1
    csi_ = pd_ + gt_
    tmp = copy.deepcopy(csi_)
    tmp[tmp == 2] = 1
    falsealarm = tmp - gt_
    if (pd_.sum()) == 0:
        return 0.0
    return float((falsealarm == 1).sum()) / float((pd_ >= 1).sum())
