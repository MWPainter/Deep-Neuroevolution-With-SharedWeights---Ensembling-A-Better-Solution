from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys

from r2r.r2r import HVG
from r2r.resblock import *
from utils import flatten





"""
Code adapted from: 
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py.
"""







__all__ = ['InceptionV4', 'inceptionv4']

pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}




def pair(v):
    if type(v) is tuple:
        if len(v) > 2:
            raise Exception("Accidentally passed something more than a pair to pair construction")
        return v
    return v,v

def conv_out_shape(spatial_shape, out_planes, kernel_size, stride, padding):
    h, w = spatial_shape
    ks = pair(kernel_size)
    st = pair(stride)
    pd = pair(padding)
    return (out_planes,
            (((h + 2*pd[0] - ks[0]) // st[0]) + 1),
            (((w + 2*pd[1] - ks[1]) // st[1]) + 1))





class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, img_shape=None):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.img_shape = img_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def _out_shape(self):
        return conv_out_shape(self.img_shape, self.conv.weight.data.size(0), self.kernel_size, self.stride, self.padding)

    def extend_hvg(self, hvg, hvn):
        hvn = hvg.add_hvn(self._out_shape(), input_modules=[self.conv], input_hvns=[hvn], batch_norm=self.bn)
        return hvg, hvn






class Mixed_3a(nn.Module):

    def __init__(self, img_shape=None):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

    def _out_shape(self):
        max_pool_out_planes = self.conv.conv.weight.data.size(1) # max pool output channels = input channels
        conv_out_planes = self.conv.conv.weight.data.size(0)
        return conv_out_shape(self.img_shape, max_pool_out_planes + conv_out_planes, kernel_size=3, stride=1, padding=0)

    def extend_hvg(self, hvg, hvn):
        hvn = hvg.add_hvn(self._out_shape(), input_modules=[self.maxpool, self.conv.conv],
                          input_hvns=[hvn, hvn], batch_norm=[None, self.conv.bn])
        return hvg, hvn





class Mixed_4a(nn.Module):

    def __init__(self, img_shape=None):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

    def _out_shape(self):
        # out shape of branch 0
        b0_out_shape = conv_out_shape(self.img_shape, self.branch0[0].conv.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        b0_out_channels, b0_out_h, b0_out_w = conv_out_shape(b0_out_shape[1:], self.branch0[1].conv.weight.data.size(0), kernel_size=3, stride=1, padding=0)
        # output channels from branch1
        b1_out_channels = self.branch1[-1].conv.weight.data.size(0)
        # Return
        return (b0_out_channels+b1_out_channels, b0_out_h, b0_out_w)

    def extend_hvg(self, hvg, hvn):
        b0_shape = conv_out_shape(self.img_shape, self.branch0[0].conv.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0[0].conv], input_hvns=[hvn], batch_norm=self.branch0[0].bn)
        b0_shape = conv_out_shape(b0_shape[1:], self.branch0[1].conv.weight.data.size(0), kernel_size=3, stride=1, padding=0)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0[1].conv], input_hvns=[b0_hvn], batch_norm=self.branch0[1].bn)

        b1_shape = conv_out_shape(self.img_shape, self.branch1[0].conv.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[0].conv], input_hvns=[hvn], batch_norm=self.branch1[0].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[1].conv.weight.data.size(0), kernel_size=(1,7), stride=1, padding=(0,3))
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[1].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[1].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[2].conv.weight.data.size(0), kernel_size=(7,1), stride=1, padding=(3,0))
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[2].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[2].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[3].conv.weight.data.size(0), kernel_size=3, stride=1, padding=0)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[3].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[3].bn)

        out_hvn = hvg.concat([b0_hvn, b1_hvn])
        return hvg, out_hvn





class Mixed_5a(nn.Module):

    def __init__(self, img_shape=None):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out

    def _out_shape(self):
        max_pool_out_planes = self.conv.conv.weight.data.size(1) # max pool output channels = input channels
        conv_out_planes = self.conv.conv.weight.data.size(0)
        return conv_out_shape(self.img_shape, max_pool_out_planes + conv_out_planes, kernel_size=3, stride=2, padding=0)

    def extend_hvg(self, hvg, hvn):
        hvn = hvg.add_hvn(self._out_shape(), input_modules=[self.conv.conv, self.maxpool],
                          input_hvns=[hvn, hvn], batch_norm=[self.conv.bn, None])
        return hvg, hvn






class Inception_A(nn.Module):

    def __init__(self, img_shape=None):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    def _out_shape(self):
        out_channels = self.branch0.conv.weight.data.size(0) + self.branch1[-1].conv.weight.data.size(0) + \
                       self.branch2[-1].conv.weight.data.size(0) + self.branch3[-1].conv.weight.data.size(0)
        return out_channels, self.img_shape[0], self.img_shape[1]

    def extend_hvg(self, hvg, hvn):
        h, w = self.img_shape

        b0_shape = (self.branch0.conv.weight.data.size(0), h, w)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0.conv], input_hvns=[hvn], batch_norm=self.branch0.bn)

        b1_shape = (self.branch1[0].conv.weight.data.size(0), h, w)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[0].conv], input_hvns=[hvn], batch_norm=self.branch1[0].bn)
        b1_shape = (self.branch1[1].conv.weight.data.size(0), h, w)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[1].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[1].bn)

        b2_shape = (self.branch2[0].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[0].conv], input_hvns=[hvn], batch_norm=self.branch2[0].bn)
        b2_shape = (self.branch2[1].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[1].conv], input_hvns=[b2_hvn], batch_norm=self.branch2[1].bn)
        b2_shape = (self.branch2[2].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[2].conv], input_hvns=[b2_hvn], batch_norm=self.branch2[2].bn)

        b3_shape = (self.branch3[1].conv.weight.data.size(1), h, w)
        b3_hvn = hvg.add_hvn(b3_shape, input_modules=[self.branch3[0]], input_hvns=[hvn], batch_norm=None)
        b3_shape = (self.branch3[1].conv.weight.data.size(0), h, w)
        b3_hvn = hvg.add_hvn(b3_shape, input_modules=[self.branch3[1].conv], input_hvns=[b3_hvn], batch_norm=self.branch3[1].bn)

        out_hvn = hvg.concat([b0_hvn, b1_hvn, b2_hvn, b3_hvn])
        return hvg, out_hvn






class Reduction_A(nn.Module):

    def __init__(self, img_shape=None):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

    def _out_shape(self):
        # out shape of branch 0
        b0_out_channels, b0_out_h, b0_out_w = conv_out_shape(self.img_shape, self.branch0.conv.weight.data.size(0), kernel_size=3, stride=2, padding=0)

        # output channels from branch1, branch2 (branch2 output channels = branch0 input channels)
        b1_out_channels = self.branch1[-1].conv.weight.data.size(0)
        b2_out_channels = self.branch0.conv.weight.data.size(1)

        return (b0_out_channels+b1_out_channels+b2_out_channels, b0_out_h, b0_out_w)

    def extend_hvg(self, hvg, hvn):
        b0_shape = conv_out_shape(self.img_shape, self.branch0.conv.weight.data.size(0), kernel_size=3, stride=2, padding=0)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0.conv], input_hvns=[hvn], batch_norm=self.branch0.bn)

        b1_shape = conv_out_shape(self.img_shape, self.branch1[0].conv.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[0].conv], input_hvns=[hvn], batch_norm=self.branch1[0].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[1].conv.weight.data.size(0), kernel_size=3, stride=1, padding=1)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[1].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[1].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[2].conv.weight.data.size(0), kernel_size=3, stride=2, padding=0)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[2].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[2].bn)

        b2_shape = conv_out_shape(self.img_shape, self.branch0.conv.weight.data.size(1), kernel_size=3, stride=2, padding=0) # branch2 output channels = branch0 input channels
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2], input_hvns=[hvn], batch_norm=None)

        out_hvn = hvg.concat([b0_hvn, b1_hvn, b2_hvn])
        return hvg, out_hvn






class Inception_B(nn.Module):

    def __init__(self, img_shape=None):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    def _out_shape(self):
        out_channels = self.branch0.conv.weight.data.size(0) + self.branch1[-1].conv.weight.data.size(0) + \
                       self.branch2[-1].conv.weight.data.size(0) + self.branch3[-1].conv.weight.data.size(0)
        return out_channels, self.img_shape[0], self.img_shape[1]

    def extend_hvg(self, hvg, hvn):
        h, w = self.img_shape

        b0_shape = (self.branch0.conv.weight.data.size(0), h, w)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0.conv], input_hvns=[hvn], batch_norm=self.branch0.bn)

        b1_shape = (self.branch1[0].conv.weight.data.size(0), h, w)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[0].conv], input_hvns=[hvn], batch_norm=self.branch1[0].bn)
        b1_shape = (self.branch1[1].conv.weight.data.size(0), h, w)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[1].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[1].bn)
        b1_shape = (self.branch1[2].conv.weight.data.size(0), h, w)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[2].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[2].bn)

        b2_shape = (self.branch2[0].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[0].conv], input_hvns=[hvn], batch_norm=self.branch2[0].bn)
        b2_shape = (self.branch2[1].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[1].conv], input_hvns=[b2_hvn], batch_norm=self.branch2[1].bn)
        b2_shape = (self.branch2[2].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[2].conv], input_hvns=[b2_hvn], batch_norm=self.branch2[2].bn)
        b2_shape = (self.branch2[3].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[3].conv], input_hvns=[b2_hvn], batch_norm=self.branch2[3].bn)
        b2_shape = (self.branch2[4].conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2[4].conv], input_hvns=[b2_hvn], batch_norm=self.branch2[4].bn)

        b3_shape = (self.branch3[1].conv.weight.data.size(1), h, w)
        b3_hvn = hvg.add_hvn(b3_shape, input_modules=[self.branch3[0]], input_hvns=[hvn], batch_norm=None)
        b3_shape = (self.branch3[1].conv.weight.data.size(0), h, w)
        b3_hvn = hvg.add_hvn(b3_shape, input_modules=[self.branch3[1].conv], input_hvns=[b3_hvn], batch_norm=self.branch3[1].bn)

        out_hvn = hvg.concat([b0_hvn, b1_hvn, b2_hvn, b3_hvn])
        return hvg, out_hvn






class Reduction_B(nn.Module):

    def __init__(self, img_shape=None):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        self.img_shape = None

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

    def _out_shape(self):
        # out shape of branch 0
        b2_out_channels, b2_out_h, b2_out_w = conv_out_shape(self.img_shape, self.branch0[0].conv.weight.data.size(1), kernel_size=3, stride=2, padding=0) # branch2 output channels = branch0 input channels

        # output channels from branch1, branch2 (branch2 output channels = branch0 input channels)
        b0_out_channels = self.branch0[-1].conv.weight.data.size(0)
        b1_out_channels = self.branch1[-1].conv.weight.data.size(1)

        return (b0_out_channels+b1_out_channels+b2_out_channels, b2_out_h, b2_out_w)

    def extend_hvg(self, hvg, hvn):
        b0_shape = conv_out_shape(self.img_shape, self.branch0[0].conv.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0[0].conv], input_hvns=[hvn], batch_norm=self.branch0[0].bn)
        b0_shape = conv_out_shape(b0_shape[1:], self.branch0[1].conv.weight.data.size(0), kernel_size=3, stride=2, padding=0)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0[1].conv], input_hvns=[b0_hvn], batch_norm=self.branch0[1].bn)

        b1_shape = conv_out_shape(self.img_shape, self.branch1[0].conv.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[0].conv], input_hvns=[hvn], batch_norm=self.branch1[0].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[1].conv.weight.data.size(0), kernel_size=(1,7), stride=1, padding=(0,3))
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[1].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[1].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[2].conv.weight.data.size(0), kernel_size=(7,1), stride=1, padding=(3,0))
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[2].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[2].bn)
        b1_shape = conv_out_shape(b1_shape[1:], self.branch1[3].conv.weight.data.size(0), kernel_size=3, stride=2, padding=0)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1[3].conv], input_hvns=[b1_hvn], batch_norm=self.branch1[3].bn)

        b2_shape = conv_out_shape(self.img_shape, self.branch0[0].conv.weight.data.size(1), kernel_size=3, stride=2, padding=0) # branch2 output channels = branch0 input channels
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2], input_hvns=[hvn], batch_norm=None)

        out_hvn = hvg.concat([b0_hvn, b1_hvn, b2_hvn])
        return hvg, out_hvn







class Inception_C(nn.Module):

    def __init__(self, img_shape=None):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

        self.img_shape = img_shape

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    def _out_shape(self):
        out_channels = self.branch0.conv.weight.data.size(0) + self.branch1_1a.conv.weight.data.size(0) + \
                       self.branch1_1b.conv.weight.data.size(0) + self.branch2_3a.conv.weight.data.size(0) + \
                       self.branch2_3b.conv.weight.data.size(0) + self.branch3[-1].conv.weight.data.size(0)
        return out_channels, self.img_shape[0], self.img_shape[1]

    def extend_hvg(self, hvg, hvn):
        h, w = self.img_shape

        b0_shape = (self.branch0.conv.weight.data.size(0), h, w)
        b0_hvn = hvg.add_hvn(b0_shape, input_modules=[self.branch0.conv], input_hvns=[hvn], batch_norm=self.branch0.bn)

        b1_shape = (self.branch1_0.conv.weight.data.size(0), h, w)
        b1_hvn = hvg.add_hvn(b1_shape, input_modules=[self.branch1_0.conv], input_hvns=[hvn], batch_norm=self.branch1_0.bn)
        b1a_shape = (self.branch1_1a.conv.weight.data.size(0), h, w)
        b1a_hvn = hvg.add_hvn(b1a_shape, input_modules=[self.branch1_1a.conv], input_hvns=[b1_hvn], batch_norm=self.branch1_1a.bn)
        b1b_shape = (self.branch1_1b.conv.weight.data.size(0), h, w)
        b1b_hvn = hvg.add_hvn(b1b_shape, input_modules=[self.branch1_1b.conv], input_hvns=[b1_hvn], batch_norm=self.branch1_1b.bn)


        b2_shape = (self.branch2_0.conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2_0.conv], input_hvns=[hvn], batch_norm=self.branch2_0.bn)
        b2_shape = (self.branch2_1.conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2_1.conv], input_hvns=[b2_hvn], batch_norm=self.branch2_1.bn)
        b2_shape = (self.branch2_2.conv.weight.data.size(0), h, w)
        b2_hvn = hvg.add_hvn(b2_shape, input_modules=[self.branch2_2.conv], input_hvns=[b2_hvn], batch_norm=self.branch2_2.bn)
        b2a_shape = (self.branch2_3a.conv.weight.data.size(0), h, w)
        b2a_hvn = hvg.add_hvn(b2a_shape, input_modules=[self.branch2_3a.conv], input_hvns=[b2_hvn], batch_norm=self.branch2_3a.bn)
        b2b_shape = (self.branch2_3b.conv.weight.data.size(0), h, w)
        b2b_hvn = hvg.add_hvn(b2b_shape, input_modules=[self.branch2_3b.conv], input_hvns=[b2_hvn], batch_norm=self.branch2_3b.bn)

        b3_shape = (self.branch3[1].conv.weight.data.size(1), h, w)
        b3_hvn = hvg.add_hvn(b3_shape, input_modules=[self.branch3[0]], input_hvns=[hvn], batch_norm=None)
        b3_shape = (self.branch3[1].conv.weight.data.size(0), h, w)
        b3_hvn = hvg.add_hvn(b3_shape, input_modules=[self.branch3[1].conv], input_hvns=[b3_hvn], batch_norm=self.branch3[1].bn)

        out_hvn = hvg.concat([b0_hvn, b1a_hvn, b1b_hvn, b2a_hvn, b2b_hvn, b3_hvn])
        return hvg, out_hvn






class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()

        # Special attributes
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None

        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),                    # (32, 149, 149)
            BasicConv2d(32, 32, kernel_size=3, stride=1),                   # (32, 147, 147)
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),        # (64, 147, 147)
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.features[0].img_shape = (299,299)
        for i in range(1,len(self.features)):
            self.features[i].img_shape = self.features[i-1]._out_shape()[1:]

        self.avg_pool = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

        self.in_shape = (3, 299, 299)
        self.out_shape = (num_classes,)

    def logits(self, features):
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    def input_shape(self):
        return self.in_shape

    def hvg(self):
        hvg = HVG(self.input_shape())
        hvg = self.conv_hvg(hvg)
        hvg = self.fc_hvg(hvg)
        return hvg

    def conv_hvg(self, cur_hvg):
        hvg = cur_hvg
        hvn = hvg.root_hvn
        for i in range(len(self.features)):
            hvg, hvn = self.features[i].extend_hvg(hvg, hvn)
        return hvg

    def fc_hvg(self, cur_hvg):
        cur_hvg.add_hvn((self.last_linear.weight.data.size(1),), input_modules=[self.avg_pool])
        cur_hvg.add_hvn(hv_shape=self.out_shape, input_modules=[self.last_linear])
        return cur_hvg






def inceptionv4(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['inceptionv4'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV4(num_classes=num_classes)
    return model







'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionv4
```
'''
if __name__ == '__main__':

    assert inceptionv4(num_classes=10, pretrained=None)
    print('success')
    assert inceptionv4(num_classes=1000, pretrained='imagenet')
    print('success')
    assert inceptionv4(num_classes=1001, pretrained='imagenet+background')
    print('success')

    # fail
    assert inceptionv4(num_classes=1001, pretrained='imagenet')