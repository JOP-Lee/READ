import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial




class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, dilation=1, padding_mode='reflect', act_fun=nn.ELU, normalization=nn.BatchNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)
        self.flag=relu

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

    def forward(self, x, *args, **kwargs):
        if self.flag:
            features = self.block.act_f(self.block.conv_f(x))
        else:
            features = self.block.conv_f(x)
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)

        return output


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(8, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-8, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out



class UNet(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.

    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        feature_scale: Division factor of number of convolutional channels. The bigger the less parameters in the model.
        num_res: Number of block resnet.
    """
    def __init__(
        self,
        num_input_channels=8, 
        num_output_channels=3,
        feature_scale=4,
        num_res=4

    ):
        super().__init__()

        self.feature_scale = feature_scale
        base_channel = 32


        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]



        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
            EBlock(base_channel*8, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(8, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel*4, base_channel*8, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*8, base_channel*4, kernel_size=4, relu=True, stride=2),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 8, num_res),
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 15, base_channel*1),
            AFF(base_channel * 15, base_channel*2),
            AFF(base_channel * 15, base_channel*4),
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.FAM0 = FAM(base_channel * 8)
        self.SCM0 = SCM(base_channel * 8)
        self.up =nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)



        n_input = len(inputs)

        x =inputs[0]
        x_2 = inputs[1]
        x_4 = inputs[2]
        x_8 = inputs[3]

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        z8 = self.SCM0(x_8)


        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)



        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        res3 = self.Encoder[2](z)

        z = self.feat_extract[6](res3)

        z = self.FAM0(z, z8)
        z = self.Encoder[3](z)



        z12 = F.interpolate(res1, scale_factor=0.5)
        z13 = F.interpolate(res1, scale_factor=0.25)

        z21 = F.interpolate(res2, scale_factor=2)
        z23 = F.interpolate(res2, scale_factor=0.5)

        z32 = F.interpolate(res3, scale_factor=2)
        z31 = F.interpolate(res3, scale_factor=4)

        z43 = F.interpolate(z, scale_factor=2)
        z42 = F.interpolate(z43, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res1 = self.AFFs[0](res1, z21, z31, z41)#AFF1
        res2 = self.AFFs[1](z12, res2, z32, z42) #AFF2
        res3 = self.AFFs[2](z13, z23, res3, z43)#AFF3


        z = self.Decoder[0](z)


        z = self.feat_extract[7](z)
        z= self.up(z)

        z = torch.cat([z, res3], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)


        z = self.feat_extract[3](z)
        z= self.up(z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)


        z = self.feat_extract[4](z)
        z= self.up(z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[2](z)
        z = self.Decoder[3](z)
        z = self.feat_extract[5](z)


        return z
