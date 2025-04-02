#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                                             ║
# ║        __  __                                        ____                __                                 ║
# ║       /\ \/\ \                                      /\  _`\             /\ \  __                            ║
# ║       \ \ \_\ \     __     _____   _____   __  __   \ \ \/\_\    ___    \_\ \/\_\    ___      __            ║
# ║        \ \  _  \  /'__`\  /\ '__`\/\ '__`\/\ \/\ \   \ \ \/_/_  / __`\  /'_` \/\ \ /' _ `\  /'_ `\          ║
# ║         \ \ \ \ \/\ \L\.\_\ \ \L\ \ \ \L\ \ \ \_\ \   \ \ \L\ \/\ \L\ \/\ \L\ \ \ \/\ \/\ \/\ \L\ \         ║
# ║          \ \_\ \_\ \__/.\_\\ \ ,__/\ \ ,__/\/`____ \   \ \____/\ \____/\ \___,_\ \_\ \_\ \_\ \____ \        ║
# ║           \/_/\/_/\/__/\/_/ \ \ \/  \ \ \/  `/___/> \   \/___/  \/___/  \/__,_ /\/_/\/_/\/_/\/___L\ \       ║
# ║                              \ \_\   \ \_\     /\___/                                         /\____/       ║
# ║                               \/_/    \/_/     \/__/                                          \_/__/        ║
# ║                                                                                                             ║
# ║           49  4C 6F 76 65  59 6F 75 2C  42 75 74  59 6F 75  4B 6E 6F 77  4E 6F 74 68 69 6E 67 2E            ║
# ║                                                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
# @Author : Lei Mou
# @File   : evaluation_metrics3D.py
import numpy as np
import SimpleITK as sitk
import glob
import os
from scipy.spatial import distance
from sklearn.metrics import f1_score
import monai
from monai.metrics import DiceMetric, MeanIoU
import torch
import torch.nn.functional as F


def numeric_score(pred, gt):
    if type(pred) is np.ndarray:
        FN = ((pred != gt) & (gt > 0)).sum()
        FP = ((pred != gt) & (gt == 0)).sum()
        TP = ((pred == gt) & (gt > 0)).sum()
        TN = ((pred == gt) & (gt == 0)).sum()
    elif type(pred) is torch.Tensor or type(pred) is monai.data.MetaTensor:
        B = pred.shape[0]
        FN = 0
        FP = 0
        TP = 0
        TN = 0
        for b in range(B):
            FN += ((pred[b] != gt[b]) & (gt[b] > 0)).sum().float()
            FP += ((pred[b] != gt[b]) & (gt[b] == 0)).sum().float()
            TP += ((pred[b] == gt[b]) & (gt[b] > 0)).sum().float()
            TN += ((pred[b] == gt[b]) & (gt[b] == 0)).sum().float()
        FN /= B
        FP /= B
        TP /= B
        TN /= B
    return FP, FN, TP, TN


def Dice_3d(pred, gt):
    iou = MeanIoU(include_background=True, reduction='mean')(pred, gt)
    dice = DiceMetric(include_background=True, reduction='mean')(pred, gt)
    return iou, dice


def IoU(pred, gt):
    pred = np.int64(pred / 255)
    gt = np.int64(gt / 255)
    m1 = np.sum(pred[gt == 1])
    m2 = np.sum(pred == 1) + np.sum(gt == 1) - m1
    iou = m1 / m2
    return iou


def metrics_3d(pred, gt):
    FP, FN, TP, TN = numeric_score(pred, gt)
    tpr = TP / (TP + FN + 1e-6)
    fnr = FN / (FN + TP + 1e-6)
    fpr = FN / (FP + TN + 1e-6)
    
    return tpr, fnr, fpr


def over_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float32(np.sum(gt == 255))
    Os = np.float32(np.sum((pred == 255) & (gt == 0)))
    OR = Os / (Rs + Os)
    return OR


def under_rate(pred, gt):
    # pred = np.int64(pred / 255)
    # gt = np.int64(gt / 255)
    Rs = np.float32(np.sum(gt == 255))
    Us = np.float32(np.sum((pred == 0) & (gt == 255)))
    Os = np.float32(np.sum((pred == 255) & (gt == 0)))
    UR = Us / (Rs + Os)
    return UR
