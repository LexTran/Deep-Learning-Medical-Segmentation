"""
3D Channel and Spatial Attention Network (CSA-Net 3D).
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import ChannelSELayer


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResEncoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        ) 
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out+residual
        out = self.relu(out)
        return out


class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels//4, in_channels//4, kernel_size=1, dilation=1)
        self.conv2 = nn.Conv3d(in_channels//4, in_channels//4, kernel_size=1, dilation=2)
        self.conv3 = nn.Conv3d(in_channels//4, in_channels//4, kernel_size=1, dilation=3)
        self.conv4 = nn.Conv3d(in_channels//4, in_channels//4, kernel_size=1, dilation=5)
        self.query = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.norm_Q = nn.LayerNorm(in_channels)
        self.norm_K = nn.LayerNorm(in_channels)
        self.norm_V = nn.LayerNorm(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        B, C, D, H, W = x.size()
        split1 = x[:, 0:C//4, ...]
        split2 = x[:, C//4:2*C//4, ...]
        split3 = x[:, 2*C//4:3*C//4, ...]
        split4 = x[:, 3*C//4:C, ...]

        split1 = self.conv1(split1)
        split2 = self.conv2(split2)
        split3 = self.conv3(split3)
        split4 = self.conv4(split4)

        split1 = split1 * self.sigmoid(split1)
        split2 = split2 * self.sigmoid(split2)
        split3 = split3 * self.sigmoid(split3)
        split4 = split4 * self.sigmoid(split4)

        x = torch.cat((split1, split2, split3, split4), dim=1)
        # compress x: [B,C,Z,H,W]-->[B,C,Z*H*W], make a matrix transpose
        Q = self.query(x)
        Q = self.norm_Q(Q.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        K = self.key(x)
        K = self.norm_K(K.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        V = self.value(x)
        V = self.norm_V(V.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        
        Q = Q.view(B, -1, D*H*W).permute(0,2,1)
        K = K.view(B, -1, D*H*W).permute(0,2,1)
        V = V.view(B, -1, D*H*W).permute(0,2,1)

        affinity = torch.bmm(Q, K.permute(0,2,1))/(C**0.5)
        affinity = self.softmax(affinity)
        affinity = torch.matmul(affinity, V)

        affinity = affinity.permute(0,2,1).view(B, C, D, H, W)
        out = self.gamma * affinity + x
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, D, H, W  = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, D, H, W)
        out = self.gamma * weights + x
        return out


class AGFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AGFF, self).__init__()
        self.GMP = nn.AdaptiveMaxPool3d(1)
        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//16, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, prev):
        feat = torch.cat((prev, x), dim=1)
        B, C, D, H, W = feat.size()
        gmp_feat = self.GMP(feat).view(B, C)
        gmp = self.MLP(gmp_feat)
        gap_feat = self.GAP(feat).view(B, C)
        gap = self.MLP(gap_feat)
        out = x*(gmp+gap).view(B,x.shape[1],1,1,1).expand_as(x)
        return prev + out


class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        # self.cab = ChannelAttentionBlock3d(in_channels)
        # self.cab = ChannelSELayer(spatial_dims=3, in_channels=in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        # cab = self.cab(x)
        out = sab + x
        return out


class MSFA(nn.Module):
    def __init__(self, in_channels=[16, 32, 64, 128], mid_channels=8, out_channels=2):
        super(MSFA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv4 = nn.Conv3d(in_channels[3], mid_channels, 1)
        self.conv3 = nn.Conv3d(in_channels[2], mid_channels, 1)
        self.attn = nn.Sequential(
            nn.Conv3d(mid_channels*2, mid_channels, 1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv3d(in_channels[1], mid_channels, 1)
        self.conv1 = nn.Conv3d(in_channels[0], mid_channels, 1)
        self.out = nn.Conv3d(mid_channels, out_channels, 1)
        

    def forward(self, D1, D2, D3, D4):
        D4 = self.conv4(D4)
        D4 = self.upsample(D4)
        D3 = self.conv3(D3)
        fuse3 = torch.cat((D3, D4), dim=1)
        D3 = D3*self.attn(fuse3)
        D3 = self.upsample(D3)
        D2 = self.conv2(D2)
        fuse2 = torch.cat((D2, D3), dim=1)
        D2 = D2*self.attn(fuse2)
        D2 = self.upsample(D2)
        D1 = self.conv1(D1)
        fuse1 = torch.cat((D1, D2), dim=1)
        D1 = D1*self.attn(fuse1)
        return self.out(D1)


class CSNet3D(nn.Module):
    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet3D, self).__init__()
        self.encoder_in = ResEncoder3d(channels, 16)
        self.encoder1 = ResEncoder3d(16, 32)
        self.encoder2 = ResEncoder3d(32, 64)
        self.encoder3 = ResEncoder3d(64, 128)
        self.encoder4 = ResEncoder3d(128, 256)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention3d(256)
        # self.attention_fuse = nn.Conv3d(256 * 2, 256, kernel_size=1)
        # self.decoder4 = Decoder3d(256, 128)
        # self.decoder3 = Decoder3d(128, 64)
        # self.decoder2 = Decoder3d(64, 32)
        # self.decoder1 = Decoder3d(32, 16)
        self.decoder4 = AGFF(256, 128)
        self.decoder3 = AGFF(128, 64)
        self.decoder2 = AGFF(64, 32)
        self.decoder1 = AGFF(32, 16)
        self.deconv4 = deconv(256, 128)
        self.deconv3 = deconv(128, 64)
        self.deconv2 = deconv(64, 32)
        self.deconv1 = deconv(32, 16)
        # self.out = nn.Conv3d(16, classes, kernel_size=1)
        self.out = MSFA(out_channels=classes)
        initialize_weights(self)

    def forward(self, x):
        enc_input = self.encoder_in(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Do Attenttion operations here
        bottleneck = self.affinity_attention(input_feature)
        # bottleneck = input_feature + attention

        # Do decoder operations here
        up4 = self.deconv4(bottleneck)
        # up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(enc3, up4)

        up3 = self.deconv3(dec4)
        # up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(enc2, up3)

        up2 = self.deconv2(dec3)
        # up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(enc1, up2)

        up1 = self.deconv1(dec2)
        # up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(enc_input, up1)
        
        final = self.out(dec1, dec2, dec3, dec4)
        return final
