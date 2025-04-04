#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: caixia_dong
@license: (C) Copyright 2020-2023, Medical Artificial Intelligence, XJTU.
@contact: caixia_dong@xjtu.edu.cn
@software: MedAI
@file: train.py
@time: 2022/7/22 14:49
@version:
@desc:
'''
import os
import os.path as osp
import sys
import datetime
import glob

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, Activations, AsDiscrete, SaveImage
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference

from model.baselines import UnetPP, VNet, UNETR, SwinUNETR
from model.csnet_3d import CSNet3D
from model.unet3d import UNet3D
from model.DSCNet import DSCNet
from dataloader.dataset import Data
from utils.evaluation_metrics3D import metrics_3d, Dice_3d
from utils.plot import visualize_3d_feat_map

parse = argparse.ArgumentParser(description="CTGAN")
parse.add_argument(
    "--data",
    type=str,
    default="301_data",
    dest="data",
    help="input data ")
parse.add_argument(
    "--model",
    type=str,
    default="UNet3D",
    dest="model",
    help="input data ")
parse.add_argument(
    "--log",
    type=str,
    default=None,
    dest="log",
    help="log path")
parse.add_argument(
    "--ckpt",
    type=str,
    default=None,
    dest="ckpt",
    help="checkpoint to use")
parse.add_argument(
    "--output",
    type=str,
    default='outputs',
    dest="output",
    help="output path")
opts = parse.parse_args()

args = {
    'data_path': f'/opt/data/nvme3/yuelele/CAS-Net/data/{opts.data}/',
    'test_shape': (128, 128, 128),
    'model_path': './save_models_randomcrop',
    'folder': f'{opts.data}_folder1',
    'model_name': opts.model,
    'num_workers': 0,
    'n_classes': 4, # seg classes, max num of label+1
    'log_path': f'./logs/{opts.log}/',
    'vis': False, # whether to visualize feature maps
    'sw_batch_size': 4
}

Test_Model = {
    'CSNet': CSNet3D,
    'UNet': UNet3D,
    'DSCNet': DSCNet,
    'Unet++': UnetPP,
    'VNet': VNet,
    'UNETR': UNETR,
    'SwinUNETR': SwinUNETR,
}

best_score = [0]
ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])

feat_maps = {}

# Tensorboard logs
writer = SummaryWriter(osp.join(args["log_path"], args['model_name'] + '_' + args['folder']))

# metrics
IoU_fn = MeanIoU(include_background=True, reduction='mean')
Dice_fn = DiceMetric(include_background=True, reduction='mean')

def hook_fn(module, input, output):
    feat_maps[module] = output


def test(net, epoch):
    print("\033[1;30;43m {} Start testing ... {}\033[0m".format("*" * 8, "*" * 8))
    net.eval()
    TP, FN, FP = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image = batch['img'].to('cuda:0')
            label = batch['label'].to('cuda:0')
            label = label.to(torch.int64)
            D = label.shape[2]
            roi_size = args['test_shape']
            sw_batch_size=args['sw_batch_size']
            pred = sliding_window_inference(image, roi_size, sw_batch_size, net)
            
            if args['vis']:
                chosen_layer = list(feat_maps.keys())[1]
                feat_map = visualize_3d_feat_map(feat_maps[chosen_layer], tb=True)
                writer.add_image('test/feature_map', feat_map, epoch, dataformats="CHW")
                
            # test metrics
            pred_class = post_pred(pred[0])
            label_class = post_label(label[0])
            tp, fn, fp = metrics_3d(pred_class.detach().cpu(), label_class.detach().cpu())
            IoU_fn([pred_class], [label_class])
            Dice_fn([pred_class], [label_class])
            TP.append(tp.detach().numpy())
            FN.append(fn.detach().numpy())
            FP.append(fp.detach().numpy())
            
            # save predictions
            pred_tensor = torch.argmax(pred, dim=1)
            output_loc = os.path.join(output_path, batch['name'][0])
            SaveImage(output_ext='')(img=pred_tensor.permute(0,2,3,1).detach().cpu(), filename=output_loc)

        iou = IoU_fn.aggregate().item()
        dice = Dice_fn.aggregate().item()
        IoU_fn.reset()
        Dice_fn.reset()
        return np.mean(TP), np.mean(FN), np.mean(FP), iou, dice


if __name__ == '__main__':
    print("______________________")
    # load dataset
    test_images = sorted(glob.glob(os.path.join(args['data_path'], "imagesTs", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args['data_path'], "labelsTs", "*.nii.gz")))
    data_dicts = [{"img": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
    test_data = Data(data_dicts, mode='test')
    test_loader = DataLoader(
        test_data, 
        batch_size=1, 
        num_workers=args['num_workers'],
        shuffle=False,
    )
    args['n_classes'] = int(test_data.n_classes)

    # create output path
    if not os.path.exists(opts.output):
        os.makedirs(opts.output)
    output_path = opts.output
    
    # post process
    if args['n_classes'] > 1:
        post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=args['n_classes'])])
    else:
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True, to_onehot=args['n_classes'])])
    post_label = AsDiscrete(to_onehot=args['n_classes'])
    
    # load weights
    epoch = opts.ckpt.split('.pkl')[0].split('-')[-1]
    net = Test_Model[args['model_name']](args['n_classes'], 1, **args).to('cuda:0') # BG is a class
    checkpoint = torch.load(opts.ckpt, map_location='cpu')
    net.load_state_dict(checkpoint)
    if args['vis']:
        net.encoder_in.register_forward_hook(hook_fn)
        net.decoder1.register_forward_hook(hook_fn)

    test(net, epoch)
