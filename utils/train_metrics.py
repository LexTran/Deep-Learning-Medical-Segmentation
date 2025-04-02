import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps, Image
from sklearn.metrics import confusion_matrix
from skimage import filters

from utils.evaluation_metrics3D import metrics_3d, Dice_3d

def metrics3d(pred, label, batch_size, train=True):
    pred_class = torch.argmax(pred, dim=1)

    tp, fn, fp, IoU, Dice = 0, 0, 0, 0, 0
    for i in range(batch_size):
        img = pred_class[i, :, :, :]
        gt = label[i, :, :, :]
        tpr, fnr, fpr = metrics_3d(img, gt)
        probs = pred[i, :, :, :]
        iou, dice = Dice_3d(probs, gt, pred.shape[1])
        tp += tpr
        fn += fnr
        fp += fpr
        IoU += iou
        Dice += dice
    return tp, fn, fp, IoU, Dice


def get_acc(image, label):
    image = threshold(image)

    FP, FN, TP, TN = numeric_score(image, label)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)
    return acc, sen
