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
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import decollate_batch
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference

from model.baselines import UnetPP, VNet, UNETR, SwinUNETR, SegResNet
from model.csnet_3d import CSNet3D
from model.unet3d import UNet3D
from model.DSCNet import DSCNet
from dataloader.dataset import Data
from utils.evaluation_metrics3D import metrics_3d, Dice_3d
from utils.model_init import init_weights
from utils.losses import WeightedCrossEntropyLoss
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
opts = parse.parse_args()

args = {
    'data_path': f'/opt/data/nvme3/yuelele/CAS-Net/data/{opts.data}/',
    'epochs': 200,
    'input_shape': (96, 128, 128),
    'snapshot': 20,
    'test_step': 10,
    'model_path': './save_models_randomcrop',
    'batch_size': 2,
    'folder': f'{opts.data}_folder1',
    'model_name': opts.model,
    'num_workers': 8,
    'n_classes': 4, # seg classes, max num of label+1
    'log_path': f'./logs/{opts.log}/',
    'vis': False, # whether to visualize feature maps
    'sw_batch_size': 4
}

Test_Model = {
    'CSNet': CSNet3D,
    'UNet': UNet3D,
    'DSCNet': DSCNet,
    'UNet++': UnetPP,
    'VNet': VNet,
    'UNETR': UNETR,
    'SwinUNETR': SwinUNETR,
    'SegResNet': SegResNet,
}

best_score = [0]
ckpt_path = os.path.join(args['model_path'], args['model_name'] + '_' + args['folder'])

feat_maps = {}

# Tensorboard logs
writer = SummaryWriter(osp.join(args["log_path"], args['model_name'] + '_' + args['folder']))

# mix scale
scaler = GradScaler()

# metrics
IoU_fn = MeanIoU(include_background=True, reduction='mean')
Dice_fn = DiceMetric(include_background=True, reduction='mean')

def hook_fn(module, input, output):
    feat_maps[module] = output


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_ckpt(net, iter):
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    date = datetime.datetime.now().strftime("%Y-%m-%d-")
    # torch.save(model.state_dict(),PATH)
    torch.save(net.state_dict(), os.path.join(ckpt_path, date + iter + '.pkl'))
    print("{} Saved model to:{}".format("\u2714", ckpt_path))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    net = Test_Model[args['model_name']](args['n_classes'], 1, **args).to('cuda:0') # BG is a class
    if opts.ckpt is None:
        init_weights(net, 'kaiming')
    else:
        checkpoint = torch.load(opts.ckpt, map_location='cpu')
        net.load_state_dict(checkpoint)

    if args['vis']:
        net.encoder_in.register_forward_hook(hook_fn)
        net.decoder1.register_forward_hook(hook_fn)
    logfile = os.path.join(ckpt_path,
                           '{}_{}_{}.txt'.format('cornary', str(args['model_name']), '128_160_160'))  # 训练日志保存地址
    sys.stdout = Logger(logfile)
    print("------------------------------------------")
    print("Network Architecture of Model unet")
    num_para = 0
    for name, param in net.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print("Number of trainable parameters {0} in Model {1}".format(num_para, str(args['model_name'])))
    print("------------------------------------------")

    # critrion = DiceFocalLoss(to_onehot_y=True, lambda_dice=0.6, lambda_focal=0.4).to('cuda:0')
    if args['n_classes'] > 1:
        critrion = DiceCELoss(include_background=True, squared_pred=True, smooth_dr=1.0e-05, to_onehot_y=True, softmax=True).to('cuda:0')
    elif args['n_classes'] == 1:
        critrion = DiceCELoss(include_background=True, squared_pred=True, smooth_dr=1.0e-05, sigmoid=True).to('cuda:0')
    critrion_wce = WeightedCrossEntropyLoss().to('cuda:0')
    # Start training
    print("\033[1;30;44m {} Start training ... {}\033[0m".format("*" * 8, "*" * 8))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    iters = 1

    for epoch in range(args['epochs']):
        scheduler.step()
        net.train()
        tp_epoch = 0
        fn_epoch = 0
        fp_epoch = 0
        loss_epoch = 0
        for idx, batch in enumerate(batchs_data):
            image = batch['img'].to('cuda:0')
            label = batch['label'].to('cuda:0')

            if image.dim() > 5:
                image = image.reshape(-1, *image.shape[2:])
                label = label.reshape(-1, *label.shape[2:])
            
            optimizer.zero_grad()

            with autocast():
                pred = net(image)
                label = label.to(torch.int64)
                loss_dice = critrion(pred, label)
                loss_wce = critrion_wce(pred, label.squeeze(1))
                loss = 0.4*loss_dice + 0.6*loss_wce
                loss = loss_dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pred_class = torch.stack([post_pred(p) for p in pred], dim=0)
            label_class = torch.stack([post_label(l.squeeze(1)) for l in label], dim=0)
            
            tp, fn, fp = metrics_3d(pred_class, label_class)
            IoU_fn([pred_class], [label_class])
            Dice_fn([pred_class], [label_class])
            tp_epoch += tp
            fn_epoch += fn
            fp_epoch += fp
            loss_epoch += loss.item()
            iters += 1
            # if args['vis'] and ((epoch + 1)%args['test_step']==0):
            #     D = label.shape[2]
            #     chosen_layer = list(feat_maps.keys())[1]
            #     feat_map = visualize_3d_feat_map(feat_maps[chosen_layer], tb=True)
            #     writer.add_image('train/image', image[0].detach().cpu().squeeze(0).numpy()[D//2,:,:], epoch, dataformats='HW')
            #     writer.add_image('train/label', label[0].detach().cpu().squeeze(0).numpy()[D//2,:,:], epoch, dataformats='HW')
            #     writer.add_image('train/feature_map', feat_map, epoch, dataformats="CHW")
            #     writer.add_image('train/pred', torch.argmax(pred[0], dim=0).detach().cpu().numpy()[D//2,:,:], epoch, dataformats="HW")
        
        iou_epoch = IoU_fn.aggregate().item()
        dice_epoch = Dice_fn.aggregate().item()
        IoU_fn.reset()
        Dice_fn.reset()
        writer.add_scalar(f'train/loss', loss_epoch/iters, epoch+1)
        writer.add_scalar(f'train/TPR', tp_epoch/len(batchs_data), epoch+1)
        writer.add_scalar(f'train/FNR', fn_epoch/len(batchs_data), epoch+1)
        writer.add_scalar(f'train/FPR', fp_epoch/len(batchs_data), epoch+1)
        writer.add_scalar(f'train/Dice', iou_epoch, epoch+1)
        writer.add_scalar(f'train/Dice', dice_epoch, epoch+1)
        print(
            '[epoch [{0:d}] loss:{1:.10f}\tTP:{2:.4f}\tFN:{3:.4f}\tFP:{4:.4f}\tIoU:{5:.4f}\tDice:{6:.4f} '.format(
                epoch + 1, loss_epoch/iters, tp_epoch / len(batchs_data), fn_epoch / len(batchs_data), fp_epoch / len(batchs_data),
                iou_epoch, dice_epoch))

        # model eval
        if (epoch + 1) % args['test_step'] == 0:
            test_tp, test_fn, test_fp, test_iou, test_dice = model_eval(net, epoch+1)
            writer.add_scalar(f'val/TPR', test_tp, epoch+1)
            writer.add_scalar(f'val/FNR', test_fn, epoch+1)
            writer.add_scalar(f'val/FPR', test_fp, epoch+1)
            writer.add_scalar(f'val/Dice', test_dice, epoch+1)
            print("Average TP:{0:.4f}, average FN:{1:.4f},  average FP:{2:.4f},  average IOU:{3:.4f}, average Dice:{4:.4f}".format(test_tp,
                                                                                                             test_fn,
                                                                                                             test_fp,
                                                                                                             test_iou,
                                                                                                             test_dice))
            if test_iou > max(best_score):
                best_score.append(test_iou)
                print(best_score)
                modelname = ckpt_path + '/' + 'best_score' + '_checkpoint.pkl'
                print('the best model will be saved at {}'.format(modelname))
                torch.save(net.state_dict(), modelname)

        if (epoch + 1) % args['snapshot'] == 0:
            save_ckpt(net, str(epoch + 1))

    print("------------------the best score of model is--------------- :", best_score)

def model_eval(net, epoch):
    print("\033[1;30;43m {} Start testing ... {}\033[0m".format("*" * 8, "*" * 8))
    net.eval()
    TP, FN, FP = [], [], []
    file_num = 0
    # val_loader.update_cache()
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            image = batch['img'].to('cuda:0')
            label = batch['label'].to('cuda:0')
            label = label.to(torch.int64)
            D = label.shape[2]
            if 'UNETR' in args['model_name']:
                roi_size = args['input_shape']
            else:
                roi_size = (128, 128, 128)
            sw_batch_size=args['sw_batch_size']
            pred_val = sliding_window_inference(image, roi_size, sw_batch_size, net)
            if args['vis']:
                chosen_layer = list(feat_maps.keys())[1]
                feat_map = visualize_3d_feat_map(feat_maps[chosen_layer], tb=True)
                writer.add_image('val/feature_map', feat_map, epoch, dataformats="CHW")
                for depth in range(D//4, 3*(D//4)):
                    writer.add_image('val/image', image.detach().cpu().squeeze(1).squeeze(0).numpy()[depth,:,:], epoch, dataformats='HW')
                    writer.add_image('val/label', label.detach().cpu().squeeze(1).squeeze(0).numpy()[depth,:,:], epoch, dataformats='HW')
                    writer.add_image('val/pred', torch.argmax(pred_val[0], dim=0).detach().cpu().numpy()[depth,:,:], epoch, dataformats="HW")

            pred_class = post_pred(pred_val[0])
            label_class = post_label(label[0])
            tp, fn, fp = metrics_3d(pred_class.detach().cpu(), label_class.detach().cpu())
            iou = IoU_fn([pred_class], [label_class])
            dice = Dice_fn([pred_class], [label_class])
            TP.append(tp.detach().numpy())
            FN.append(fn.detach().numpy())
            FP.append(fp.detach().numpy())
            file_num += 1
        iou = IoU_fn.aggregate().item()
        dice = Dice_fn.aggregate().item()
        IoU_fn.reset()
        Dice_fn.reset()
        return np.mean(TP), np.mean(FN), np.mean(FP), iou, dice


if __name__ == '__main__':
    print("______________________")
    # load train dataset
    train_images = sorted(glob.glob(os.path.join(args['data_path'], "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args['data_path'], "labelsTr", "*.nii.gz")))
    data_dicts = [{"img": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-int(np.ceil(len(train_images)*0.1))], data_dicts[-int(np.ceil(len(train_images)*0.1)):]
    train_data = Data(train_files, input_shape=args['input_shape'])
    batchs_data = DataLoader(
        train_data, 
        batch_size=args['batch_size'], 
        num_workers=args['num_workers'], 
        shuffle=True,
        drop_last=True,
    )
    val_data = Data(val_files, mode='test')
    val_loader = DataLoader(
        val_data, 
        batch_size=1, 
        num_workers=args['num_workers'],
    )
    args['n_classes'] = int(train_data.n_classes)
    # post process
    if args['n_classes'] > 1:
        post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=args['n_classes'])])
    else:
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True, to_onehot=args['n_classes'])])
    post_label = AsDiscrete(to_onehot=args['n_classes'])
    train()
