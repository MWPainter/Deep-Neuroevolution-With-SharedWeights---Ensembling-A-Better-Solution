import numpy as np
import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo

from r2r.r2r import *
from r2r.net2net import *
from r2r.init_utils import _extend_filter_with_repeated_in_channels, _extend_filter_with_repeated_out_channels
from r2r.module_utils import _assign_kernel_and_bias_to_conv_, _assign_to_batch_norm_
from r2r.residual_connection import Residual_Connection


"""
Code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""





__all__ = ['ResNet', 'resnet10', 'resnet10_cifar', 'resnet18', 'resnet18_cifar', 'resnet34', 'resnet26', 'resnet26_r2r', 'resnet50', 'resnet101',
           'resnet152']





model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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




def _identity_init_conv(conv_layer, noise=True):
    """ Identity inits a convolutional filter. """
    conv_kernel = conv_layer.weight.data.cpu()
    conv_kernel.zero_()
    if conv_layer.bias is not None:
        conv_bias = conv_layer.bias.data.cpu()
        conv_bias.zero_()

    out_channels, in_channels, height, width = conv_kernel.shape
    center_height = (height - 1) // 2
    center_width = (width - 1) // 2

    for i in range(0, out_channels):
        conv_kernel.narrow(0, i, 1).narrow(1, i, 1).narrow(2, center_height, 1).narrow(3, center_width, 1).fill_(1)

    if noise:
        noise = np.random.normal(scale=5e-5, size=list(conv_kernel.size()))
        conv_kernel += t.FloatTensor(noise).type_as(conv_kernel)

    new_weights = conv_kernel.numpy()
    new_bias = None if conv_layer.bias is None else conv_bias.numpy()

    _assign_kernel_and_bias_to_conv_(conv_layer, new_weights, new_bias)




def _identity_init_batch_norm(batch_norm, entire_resnet, batch):
    """ Identity inits a batch norm, using the batch statistics from 'batch'. """
    bn_momentum = batch_norm.momentum
    batch_norm.reset_running_stats()
    batch_norm.momentum = 1.0
    entire_resnet(batch)
    batch_norm.momentum = bn_momentum
    new_weights = np.sqrt(batch_norm.running_var.numpy() + batch_norm.eps)
    _assign_to_batch_norm_(batch_norm, new_weights, batch_norm.running_mean.numpy(),
                           batch_norm.running_mean.numpy(), batch_norm.running_var.numpy())





def reduce_size_function(ratio):
    """Reduces by ratio r, to give an interger, and force it to be a multiple of eight"""
    return lambda x: int((((x // ratio) + 7) // 8) * 8)





def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)





def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, identity_initialize=False, img_shape=None,
                 use_residual=True, use_net2net=False, add_noise=True, init_scale=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.res = Residual_Connection()
        self.img_shape = img_shape
        self.use_residual = use_residual

        # scaled inits for params if init_scale is not none
        if init_scale is not None:
            bound = np.sqrt(3.0) * init_scale
            conv1_init = np.random.uniform(-bound, bound, size=(inplanes, planes, 3, 3)).astype(np.float32)
            # conv1_init = init_scale * np.random.randn(inplanes, planes, 3, 3).astype(np.float32)
            _assign_kernel_and_bias_to_conv_(self.conv1, conv1_init)
            bound = np.sqrt(3.0) * init_scale
            conv2_init = np.random.uniform(-bound, bound, size=(planes, planes, 3, 3)).astype(np.float32)
            # conv2_init = init_scale * np.random.randn(planes, planes, 3, 3).astype(np.float32)
            _assign_kernel_and_bias_to_conv_(self.conv2, conv2_init)

        # R2DeeperR
        if identity_initialize and not use_net2net:
            # When deepening, we shouldn't decrease the spatial dimension (for now at least
            if self.downsample:
                raise Exception("Can't deepen/identity initialize a residual block when it's downsampling spatial dimensions")

            # Initialize the conv weights as appropriate, using the helpers
            conv1_filter_shape = (planes, inplanes, 3, 3)
            init_type = 'He' if init_scale is None else 'scale'
            conv1_filter_init = _extend_filter_with_repeated_out_channels(conv1_filter_shape, init_type=init_type, std=init_scale)
            _assign_kernel_and_bias_to_conv_(self.conv1, conv1_filter_init)

            conv2_filter_shape = (planes * self.expansion, planes, 3, 3)
            conv2_filter_init = _extend_filter_with_repeated_in_channels(conv2_filter_shape, init_type=init_type, std=init_scale, alpha=-1.0)
            _assign_kernel_and_bias_to_conv_(self.conv2, conv2_filter_init)

            # Initialize the batch norm variables so that the scale is one and the mean is zero
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)

        # Net2DeeperNet - note that the batch norms need to be edited in the deepening after
        elif identity_initialize and use_net2net:
            # Error checking
            if use_residual:
                raise Exception("Cannot identity initialize a res block when using it's residual connection.")

            # Identity init all of the convolutions
            _identity_init_conv(self.conv1, noise=add_noise)
            _identity_init_conv(self.conv2, noise=add_noise)


    def _net2deepernet_batch_norm_correction(self, entire_resnet, batch):
        """
        In net2deepernet we need to set parameters in the batch norm so that it "undoes" the normalization. We need
        to iterate through the layers and do that for each layer.

        :param entire_resnet: The nn.Module for the entire resenet, so we can run the forward pass (including this
                resblock)
        :param batch: A minibatch to run through the network.
        """
        _identity_init_batch_norm(self.bn1, entire_resnet, batch)
        _identity_init_batch_norm(self.bn2, entire_resnet, batch)



    def _get_conv_scale(self):
        """
        Get the std of the kernel of the last layer in this block
        """
        return np.std(self.conv2.weight.detach().cpu().numpy())



    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_residual:
            out = self.res(out, identity)
        out = self.relu(out)

        return out


    def _out_shape(self):
        """
        Output shape takes into account the spatial reduction from conv2, and sets the out channels using conv3
        :return:
        """
        return conv_out_shape(self.img_shape, self.conv2.weight.data.size(0), kernel_size=3, stride=self.stride, padding=1)


    def extend_hvg(self, hvg, hvn):
        conv1_shape = conv_out_shape(self.img_shape, self.conv1.weight.data.size(0), kernel_size=3, stride=self.stride, padding=1)
        conv1_hvn = hvg.add_hvn(conv1_shape, input_modules=[self.conv1], input_hvns=[hvn], batch_norm=self.bn1)

        conv2_shape = conv_out_shape(conv1_shape[1:], self.conv2.weight.data.size(0), kernel_size=3, stride=1, padding=1)
        conv2_hvn = hvg.add_hvn(conv2_shape, input_modules=[self.conv2], input_hvns=[conv1_hvn], batch_norm=self.bn2)

        if self.downsample:
            ds_shape = conv_out_shape(self.img_shape, self.downsample[0].weight.data.size(0), kernel_size=3, stride=self.stride, padding=1)
            ds_hvn = hvg.add_hvn(ds_shape, input_modules=[self.downsample[0]], input_hvns=[hvn], batch_norm=self.downsample[1], residual_connection=self.res)
        else:
            hvn.residual_connection = self.res

        return hvg, conv2_hvn





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, identity_initialize=False, img_shape=None,
                 use_residual=True, use_net2net=False, add_noise=True, init_scale=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.res = Residual_Connection()
        self.img_shape = img_shape
        self.use_residual = use_residual

        # scaled inits for params if init_scale is not none
        if init_scale is not None:
            # conv1_init = init_scale * np.random.randn(inplanes, planes, 1, 1).astype(np.float32)
            bound = np.sqrt(3.0) * init_scale
            conv1_init = np.random.uniform(-bound, bound, size=(inplanes, planes, 1, 1)).astype(np.float32)
            _assign_kernel_and_bias_to_conv_(self.conv1, conv1_init)
            # conv2_init = init_scale * np.random.randn(planes, planes, 3, 3).astype(np.float32)
            bound = np.sqrt(3.0) * init_scale
            conv2_init = np.random.uniform(-bound, bound, size=(planes, planes, 3, 3)).astype(np.float32)
            _assign_kernel_and_bias_to_conv_(self.conv2, conv2_init)
            # conv3_init = init_scale * np.random.randn(planes, planes * self.expansion, 1, 1).astype(np.float32)
            bound = np.sqrt(3.0) * init_scale
            conv3_init = np.random.uniform(-bound, bound, size=(planes, planes * self.expansion, 1, 1)).astype(np.float32)
            _assign_kernel_and_bias_to_conv_(self.conv3, conv3_init)

        # R2DeeperR
        if identity_initialize and not use_net2net:
            # When deepening, we shouldn't decrease the spatial dimension (for now at least
            if self.downsample:
                raise Exception("Can't deepen/identity initialize a residual block when it's downsampling spatial dimensions")

            # Initialize the conv weights as appropriate, using the helpers
            conv2_filter_shape = (planes, planes, 3, 3)
            init_type = 'He' if init_scale is None else 'scale'
            conv2_filter_init = _extend_filter_with_repeated_out_channels(conv2_filter_shape, init_type=init_type, std=init_scale)
            self.conv2.weight.data = Parameter(t.Tensor(conv2_filter_init))

            conv3_filter_shape = (planes * self.expansion, planes, 1, 1)
            conv3_filter_init = _extend_filter_with_repeated_in_channels(conv3_filter_shape, init_type=init_type, std=init_scale, alpha=-1.0)
            self.conv3.weight.data = Parameter(t.Tensor(conv3_filter_init))

            # Initialize the batch norm variables so that the scale is one and the mean is zero
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            nn.init.constant_(self.bn2.weight, 1)
            nn.init.constant_(self.bn2.bias, 0)
            nn.init.constant_(self.bn3.weight, 1)
            nn.init.constant_(self.bn3.bias, 0)

        # Net2DeeperNet - note that the batch norms need to be edited in the deepening after
        elif identity_initialize and use_net2net:
            # Error checking
            if use_residual:
                raise Exception("Cannot identity initialize a res block when using it's residual connection.")

            # Identity init all of the convolutions
            _identity_init_conv(self.conv1, noise=add_noise)
            _identity_init_conv(self.conv2, noise=add_noise)
            _identity_init_conv(self.conv3, noise=add_noise)


    def _net2deepernet_batch_norm_correction(self, entire_resnet, batch):
        """
        In net2deepernet we need to set parameters in the batch norm so that it "undoes" the normalization. We need
        to iterate through the layers and do that for each layer.

        :param entire_resnet: The nn.Module for the entire resenet, so we can run the forward pass (including this
                resblock)
        :param batch: A minibatch to run through the network.
        """
        _identity_init_batch_norm(self.bn1, entire_resnet, batch)
        _identity_init_batch_norm(self.bn2, entire_resnet, batch)
        _identity_init_batch_norm(self.bn3, entire_resnet, batch)



    def _get_conv_scale(self):
        """
        Get the std of the kernel of the last layer in this block
        """
        return np.std(self.conv3.weight.detach().cpu().numpy())


    def forward(self, x):
        """
        If downsample, then x2 reduces the spatial dimensions, as does "identity":
        x -> x1 -> x2 -> x3 -> + -> out
        |------> identity -----^

        If not downsample:
        x -> x1 -> x2 -> x3 -> + -> out
        |----------------------^
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_residual:
            out = self.res(out, identity)
        out = self.relu(out)

        return out

    def _out_shape(self):
        """
        Output shape takes into account the spatial reduction from conv2, and sets the out channels using conv3
        :return:
        """
        return conv_out_shape(self.img_shape, self.conv3.weight.data.size(0), kernel_size=3, stride=self.stride, padding=1)

    def extend_hvg(self, hvg, hvn):
        conv1_shape = conv_out_shape(self.img_shape, self.conv1.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        conv1_hvn = hvg.add_hvn(conv1_shape, input_modules=[self.conv1], input_hvns=[hvn], batch_norm=self.bn1)

        conv2_shape = conv_out_shape(conv1_shape[1:], self.conv2.weight.data.size(0), kernel_size=3, stride=self.stride, padding=1)
        conv2_hvn = hvg.add_hvn(conv2_shape, input_modules=[self.conv2], input_hvns=[conv1_hvn], batch_norm=self.bn2)

        conv3_shape = conv_out_shape(conv2_shape[1:], self.conv3.weight.data.size(0), kernel_size=1, stride=1, padding=0)
        conv3_hvn = hvg.add_hvn(conv3_shape, input_modules=[self.conv3], input_hvns=[conv2_hvn], batch_norm=self.bn3)

        if self.downsample:
            ds_shape = conv_out_shape(self.img_shape, self.downsample[0].weight.data.size(0), kernel_size=3, stride=self.stride, padding=1)
            ds_hvn = hvg.add_hvn(ds_shape, input_modules=[self.downsample[0]], input_hvns=[hvn], batch_norm=self.downsample[1], residual_connection=self.res)
        else:
            hvn.residual_connection = self.res

        return hvg, conv3_hvn





class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, r=(lambda x: x),
                 function_preserving=True, use_residual=True, morphism_scheme="r2r", init_scheme='match_std'):
        super(ResNet, self).__init__()

        self.block = block

        self.function_preserving = function_preserving
        self.approx_widened_ratio = 1.0
        self.in_shape = (3,224,224)
        self.out_shape = (num_classes,)
        self.r = r
        self.use_residual = use_residual
        self.morphism_scheme = morphism_scheme.lower()
        self.init_scheme = init_scheme

        self.inplanes = r(64)
        self.conv1 = nn.Conv2d(3, r(64), kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(r(64))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        conv1_out_shape = conv_out_shape(self.in_shape[1:], self.conv1.weight.data.size(0), kernel_size=7, stride=2, padding=3)
        pool_out_shape = conv_out_shape(conv1_out_shape[1:], self.conv1.weight.data.size(0), kernel_size=3, stride=2, padding=1)
        self.layer1_modules, out_img_shape = self._make_layer(block, r(64), pool_out_shape[1:], layers[0])
        self.layer2_modules, out_img_shape = self._make_layer(block, r(128), out_img_shape, layers[1], stride=2)
        self.layer3_modules, out_img_shape = self._make_layer(block, r(256), out_img_shape, layers[2], stride=2)
        self.layer4_modules, out_img_shape = self._make_layer(block, r(512), out_img_shape, layers[3], stride=2)
        self.layer1 = nn.Sequential(*self.layer1_modules)
        self.layer2 = nn.Sequential(*self.layer2_modules)
        self.layer3 = nn.Sequential(*self.layer3_modules)
        self.layer4 = nn.Sequential(*self.layer4_modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(r(512) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, img_shape, num_blocks, stride=1):
        if num_blocks == 0:
            return [], img_shape

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, img_shape=img_shape, use_residual=self.use_residual))
        img_shape = conv_out_shape(img_shape, planes, kernel_size=3, stride=stride, padding=1)[1:]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, img_shape=img_shape, use_residual=self.use_residual))
        return nn.ModuleList(layers), img_shape


    def _deepen_layer(self, layer_modules, block, num_blocks, minibatch=None, add_noise=True):
        # From the last block in this 'layer' of the resnet, work out the correct planes/inplanes and img shape for a new block
        inplanes, h, w = layer_modules[-1]._out_shape()
        planes = inplanes // block.expansion

        # Set an init scale. If it's a number, then initialize the blocks with noise with that stddev. None => use He init
        init_scale = layer_modules[-1]._get_conv_scale() * 1.0e-1 if self.function_preserving else None

        # Add the new block
        for _ in range(num_blocks):
            use_net2net_deepening = (self.morphism_scheme in ["net2net", "netmorph"])
            identity_block = block(inplanes, planes, identity_initialize=self.function_preserving, img_shape=(h,w),
                                   use_residual=self.use_residual, use_net2net=use_net2net_deepening,
                                   add_noise=add_noise, init_scale=init_scale)
            if use_net2net_deepening:
                identity_block._net2deepernet_batch_norm_correction(self, minibatch)
            layer_modules.append(identity_block)



    def deepen(self, num_blocks, minibatch=None, add_noise=True):
        self._deepen_layer(self.layer1_modules, self.block, num_blocks[0], minibatch, add_noise)
        self._deepen_layer(self.layer2_modules, self.block, num_blocks[1], minibatch, add_noise)
        if len(self.layer3_modules) > 0 and len(self.layer4_modules) > 0:
            self._deepen_layer(self.layer3_modules, self.block, num_blocks[2], minibatch, add_noise)
            self._deepen_layer(self.layer4_modules, self.block, num_blocks[3], minibatch, add_noise)
        elif len(num_blocks) > 2 and (num_blocks[2] > 0 or num_blocks[3] > 0):
            raise Exception("Cannot deepen on spatial stacks that don't existing in the resnet.")
        print()
        print(len(self.layer1_modules))
        print(len(self.layer2_modules))
        print(len(self.layer3_modules))
        print(len(self.layer4_modules))
        print(len(num_blocks))
        print(num_blocks[2])
        print(num_blocks[3])
        self.layer1 = nn.Sequential(*self.layer1_modules)
        self.layer2 = nn.Sequential(*self.layer2_modules)
        self.layer3 = nn.Sequential(*self.layer3_modules)
        self.layer4 = nn.Sequential(*self.layer4_modules)



    def widen(self, ratio):
        if not self.morphism_scheme == "net2net":
            use_network_morphism_scheme = self.morphism_scheme == "netmorph"
            widen_network_(self, new_channels=ratio, new_hidden_nodes=ratio, init_type=self.init_scheme,
                           function_preserving=self.function_preserving, multiplicative_widen=True, mfactor=8,
                           net_morph=use_network_morphism_scheme)
        else:
            net2net_widen_network_(self, new_channels=ratio, new_hidden_nodes=ratio, multiplicative_widen=True,
                                   add_noise=True, mfactor=8)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def input_shape(self):
        return self.in_shape

    def hvg(self):
        hvg = HVG(self.input_shape())
        hvg = self.conv_hvg(hvg)
        hvg = self.fc_hvg(hvg)
        return hvg

    def conv_hvg(self, cur_hvg):
        conv1_out_shape = conv_out_shape(self.in_shape[1:], self.conv1.weight.data.size(0), kernel_size=7, stride=2, padding=3)
        conv1_hvn = cur_hvg.add_hvn(conv1_out_shape, input_modules=[self.conv1], batch_norm=self.bn1)
        pool_out_shape = conv_out_shape(conv1_out_shape[1:], self.conv1.weight.data.size(0), kernel_size=3, stride=2, padding=1)
        cur_hvn = cur_hvg.add_hvn(pool_out_shape, input_modules=[self.maxpool], input_hvns=[conv1_hvn])

        for block in self.layer1_modules:
            cur_hvg, cur_hvn = block.extend_hvg(cur_hvg, cur_hvn)
        for block in self.layer2_modules:
            cur_hvg, cur_hvn = block.extend_hvg(cur_hvg, cur_hvn)
        for block in self.layer3_modules:
            cur_hvg, cur_hvn = block.extend_hvg(cur_hvg, cur_hvn)
        for block in self.layer4_modules:
            cur_hvg, cur_hvn = block.extend_hvg(cur_hvg, cur_hvn)

        conv_out_channels = cur_hvn.hv_shape[0]
        cur_hvn = cur_hvg.add_hvn((conv_out_channels, 1, 1), input_modules=[self.avgpool], input_hvns=[cur_hvn])

        return cur_hvg

    def fc_hvg(self, cur_hvg):
        cur_hvg.add_hvn(hv_shape=self.out_shape, input_modules=[self.fc])
        return cur_hvg





def resnet10(thin=False, thinning_ratio=2, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(BasicBlock, [1, 1, 1, 1], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    return model



def resnet10_cifar(thin=False, thinning_ratio=2, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-10 model for cifar.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(BasicBlock, [2, 2, 0, 0], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    return model



def resnet18_cifar(pretrained=False, thin=False, thinning_ratio=2, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(BasicBlock, [4, 4, 0, 0], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    return model



def resnet18(pretrained=False, thin=False, thinning_ratio=2, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(BasicBlock, [2, 2, 2, 2], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, thin=False, thinning_ratio=2, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(BasicBlock, [3, 4, 6, 3], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet26(thin=False, thinning_ratio=4, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(Bottleneck, [2, 2, 2, 1], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    return model



def resnet26_r2r(thin=False, thinning_ratio=4, morphism_scheme="r2r"):
    model = resnet26(thin=True, thinning_ratio=thinning_ratio, morphism_scheme=morphism_scheme)
    if not thin:
        model.widen(thinning_ratio)
    return model




def resnet50(pretrained=False, thin=False, thinning_ratio=4, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(Bottleneck, [3, 4, 6, 3], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet50_r2r(thin=False, thinning_ratio=4):
    model = resnet50(thin=True, thinning_ratio=thinning_ratio)
    if not thin:
        model.widen(thinning_ratio)
    return model





def resnet101(pretrained=False, thin=False, thinning_ratio=4, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(Bottleneck, [3, 4, 23, 3], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model





def resnet152(pretrained=False, thin=False, thinning_ratio=4, function_preserving=True, use_residual=True, morphism_scheme="r2r", **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    r = lambda x: x
    if thin:
        r = reduce_size_function(thinning_ratio)
    model = ResNet(Bottleneck, [3, 8, 36, 3], function_preserving=function_preserving, r=r, use_residual=use_residual, morphism_scheme=morphism_scheme, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model