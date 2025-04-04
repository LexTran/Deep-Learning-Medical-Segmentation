from __future__ import print_function, division
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import warnings
import numpy as np
from scipy import ndimage
import nibabel as nib
from monai.transforms import LoadImaged, RandSpatialCropd, Compose, Orientationd
from monai.transforms import RandAffined, RandFlipd, RandGaussianNoised
from monai.transforms import CropForegroundd, RandCropByPosNegLabeld, ScaleIntensityd


warnings.filterwarnings('ignore')


def load_dataset(root_dir, folder, train=True):
    images_path = os.path.join(root_dir, 'img')
    groundtruth_path = os.path.join(root_dir, 'mask')
    print("folder", folder)
    # images_path = os.path.join(root_dir,  'img')
    # groundtruth_path = os.path.join(root_dir, 'mask')
    # i=0
    folder_file = './datasets/' + folder

    # if self.train_type in ['train', 'validation', 'test']:
    if train:
        # this is for cross validation
        with open(os.path.join(folder_file, folder_file.split('/')[-1] + '_' + 'train' + '.list'),
                  'r') as f:
            image_list = f.readlines()
    else:
        with open(os.path.join(folder_file, folder_file.split('/')[-1] + '_' + 'validation' + '.list'),
                  'r') as f:
            image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]
    images = [os.path.join(root_dir, 'imagesTr', x) for x in image_list]
    groundtruth = [os.path.join(root_dir, 'labelsTr', x.replace('.img', '.label')) for x in image_list]
    return images, groundtruth


def resize3d(img3d, reshape):
    zoom_seq = np.array(reshape, dtype='float') / np.array(img3d.shape, dtype='float')
    ret = ndimage.interpolation.zoom(img3d, zoom_seq, order=1, prefilter=False)
    return ret.astype(img3d.dtype)


class Data(Dataset):
    def __init__(self,
                 dict_list,
                 input_shape=None,
                 num_samples=4,
                 mode='train',
                 device=torch.device('cpu'),
                 train=True,
                 rotate=40,
                 flip=True,
                 scale1=512):

        self.dict_list = dict_list 
        # self.images, self.groundtruth = load_dataset(self.root_dir, self.folder, self.train)
        self.loader = Compose([
            LoadImaged(keys=["img", "label"], ensure_channel_first=True),
            Orientationd(keys=["img", "label"], axcodes="RAS"),
        ])
        max_label = self.loader(dict_list[0])['label'].max()
        if max_label == 255:
            max_label = max_label//255
        self.n_classes = max_label+1
        self.mode = mode
        if mode == 'train':
            self.aug = Compose([
                ScaleIntensityd(keys=["img"]),
                CropForegroundd(keys=["img", "label"], source_key="img"),
                RandCropByPosNegLabeld(
                    keys=["img", "label"],
                    label_key="label",
                    spatial_size=input_shape,
                    pos=1,
                    neg=1,
                    num_samples=num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandAffined(
                    keys=['img', 'label'], 
                    prob=0.5, 
                    mode=('bilinear', 'nearest'),
                    translate_range=(40,40,2),
                    rotate_range=(np.pi/36, np.pi/36, np.pi/4),
                    scale_range=(0.15,0.15,0.15),
                    padding_mode='border',
                ),
                RandFlipd(keys=['img', 'label'], prob=0.2, spatial_axis=[0]),
                RandFlipd(keys=['img', 'label'], prob=0.2, spatial_axis=[1]),
                RandFlipd(keys=['img', 'label'], prob=0.2, spatial_axis=[2]),
                RandGaussianNoised(keys=['img'], prob=0.2),   
            ])
        elif mode == 'test':
            self.aug = Compose([  
                ScaleIntensityd(keys=["img"]),
                CropForegroundd(keys=["img", "label"], source_key="img"),
            ])

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self, idx):
        name = self.dict_list[idx]['img'].split('/')[-1]
        data = self.loader(self.dict_list[idx])

        # use for debug: check the coordinate correspondence
        # from PIL import Image
        # img_patch = data['img'][0]
        # label_patch = data['label'][0]
        # img_patch = img_patch.detach().cpu().numpy()
        # img_patch = img_patch.astype(np.uint8)
        # Image.fromarray(img_patch[:,:,128]).save('img.png', format='png')
        
        # label_patch = label_patch.astype(torch.int64)
        # label_patch = torch.nn.functional.one_hot(label_patch, num_classes=4)
        # label_patch *= 255
        # label_patch = label_patch.detach().cpu().numpy()
        # label_patch = label_patch.astype(np.uint8)
        # Image.fromarray(label_patch[:,:,128,:]).convert('RGB').save('label.png', format='png')

        # ensure z first
        data['label'] = data['label'].permute(0,3,1,2).astype(np.int64)
        data['img'] = data['img'].permute(0,3,1,2)

        if data['label'].max() == 255:
            data['label'] = data['label']//255

        if self.mode == 'train':
            data_list = self.aug(data)
            img_list = [data['img'] for data in data_list]
            label_list = [data['label'] for data in data_list]
            image = torch.stack(img_list, dim=0)
            label = torch.stack(label_list, dim=0)

        elif self.mode == 'test':
            data = self.aug(data)
            image = data['img']
            label = data['label']

        return {'img':image, 'label':label, 'name':name}


def train_transform(spatial_size, num_samples=4):
    train_transforms = Compose([
        LoadImaged(keys=["img", "label"], ensure_channel_first=True),
        ScaleIntensityd(keys=["img"]),
        CropForegroundd(keys=["img", "label"], source_key="img"),
        RandCropByPosNegLabeld(
            keys=["img", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandAffined(
            keys=['img', 'label'], 
            prob=0.5, 
            mode=('bilinear', 'nearest'),
            translate_range=(40,40,2),
            rotate_range=(np.pi/36, np.pi/36, np.pi/4),
            scale_range=(0.15,0.15,0.15),
            padding_mode='border',
        ),
        RandFlipd(keys=['img', 'label'], prob=0.2, spatial_axis=[0]),
        RandFlipd(keys=['img', 'label'], prob=0.2, spatial_axis=[1]),
        RandFlipd(keys=['img', 'label'], prob=0.2, spatial_axis=[2]),
        RandGaussianNoised(keys=['img'], prob=0.2),   
    ])
    return train_transforms

val_transform = Compose([
    LoadImaged(keys=["img", "label"], ensure_channel_first=True),
    ScaleIntensityd(keys=["img"]),
    CropForegroundd(keys=["img", "label"], source_key="img"),   
])