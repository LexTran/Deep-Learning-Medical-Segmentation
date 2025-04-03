import torch.nn as nn
from monai.networks import nets

class UnetPP(nn.Module):
    def __init__(self, classes, channels, **opt):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(UnetPP, self).__init__()
        self.model = nets.BasicUnetPlusPlus(in_channels=channels, out_channels=classes)

    def forward(self, x):
        return self.model(x)[0]


class VNet(nn.Module):
    def __init__(self, classes, channels, **opt):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(VNet, self).__init__()
        self.model = nets.VNet(in_channels=channels, out_channels=classes)

    def forward(self, x):
        return self.model(x)


class UNETR(nn.Module):
    def __init__(self, classes, channels, **opt):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(UNETR, self).__init__()
        self.model = nets.UNETR(in_channels=channels, out_channels=classes, img_size=opt['input_shape'])

    def forward(self, x):
        return self.model(x)


class SwinUNETR(nn.Module):
    def __init__(self, classes, channels, **opt):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(SwinUNETR, self).__init__()
        self.model = nets.SwinUNETR(in_channels=channels, out_channels=classes, img_size=opt['input_shape'], feature_size=48, use_checkpoint=True)

    def forward(self, x):
        return self.model(x)


class SegResNet(nn.Module):
    def __init__(self, classes, channels, **opt):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(SegResNet, self).__init__()
        self.model = nets.SegResNetDS(in_channels=channels, out_channels=classes, resolution=opt['input_shape'])

    def forward(self, x):
        return self.model(x)