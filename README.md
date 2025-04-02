# https://github.com/Cassie-CV/CAS-NetCoronary Artery Segmentation Nets

This repo is designed for integration of common coronary artery segmentation networks (I have only test it on coronary artery CTA data, but should work for most CT data), you can modify it to fit your data, that should be easy.

## Train

please make sure your data is organized as follows:

```
--data
----imagesTr
------xxx.nii.gz
------xxx.nii.gz
----labelsTr
------xxx.nii.gz
------xxx.nii.gz
----imagesTs
------xxx.nii.gz
------xxx.nii.gz
----labelsTs
------xxx.nii.gz
------xxx.nii.gz
```

Out of personal preference, I'd like to save the data as `nii.gz` format, but you may change the dataset to load `.npy` or whatever format you like, just modify following code in 'dataloader/npy_3d_Loader.py':

```python
65        self.loader = Compose([
66            LoadImaged(keys=["img", "label"], ensure_channel_first=True),
67            Orientationd(keys=["img", "label"], axcodes="RAS"),
68        ])
```

The `LoadImaged` function should be work for most of the format, but just in case so you know how to modify it.

Once you have placed your data correctly, you are ready to train the nets. I have provided you a `bash` script for you to easily run the training. Simply type following command in your terminal and you are training the net!

```shell
sh train.sh
```

If you want to change the net you are training, you can simply change the model name in the training script.

I have split the training set into training and validation sets at a ratio of 9:1 in train.py, so you don't have to split validation set yourself, but this would cause the training process slightly different each time you run it, it is totally normal, so don't panic~

## Features

- [X] UNet
- [X] CAS-Net
- [X] Feature map visualization
- [X] Continue training from checkpoint
- [X] In-training results visualized logs

## TODO

- [ ] test script
- [ ] test MONAI-based nets
- [ ] DSCNet
- [ ] SSL

## Acknowledgements

This repo is an unofficial Pytorch-based implementation of Coronary Artery Segmentation in CTA data and highly based on [MONAI](https://docs.monai.io/en/stable/) and [CAS-Net](https://github.com/Cassie-CV/CAS-Net). This project is not perfect. If you are using this repo and encounter with some issues, please let me know.
