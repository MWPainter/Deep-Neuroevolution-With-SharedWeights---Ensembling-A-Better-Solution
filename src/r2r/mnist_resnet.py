import torch as t
import torch.nn as nn
import torch.nn.functional as F

from r2r.r2r import HVG
from r2r.resblock import *
from utils import flatten

from itertools import chain





__all__ = ['Mnist_Resnet']





class Mnist_Resnet(nn.Module):
    """
    A small residual network to be used for mnist tests.

    Implements the R2R interface.
    """
    def __init__(self, identity_initialize=True, add_residual=True):
        # Superclass initializer
        super(Mnist_Resnet, self).__init__()
        
        # Make the three conv layers, with three max pools
        self.resblock1 = Res_Block(input_channels=1, intermediate_channels=[8,8,8], 
                                   output_channels=8, identity_initialize=identity_initialize,
                                   input_spatial_shape=(32,32), add_residual=add_residual)       # [-1, 8, 32, 32]
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 8, 16, 16]    
        self.resblock2 = Res_Block(input_channels=8, intermediate_channels=[16,16,16],
                                   output_channels=16, identity_initialize=identity_initialize,
                                   input_spatial_shape=(16,16), add_residual=add_residual)       # [-1, 16, 16, 16]
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 16, 8, 8]
        self.resblock3 = Res_Block(input_channels=16, intermediate_channels=[32,32,32],
                                   output_channels=32, identity_initialize=identity_initialize,
                                   input_spatial_shape=(8,8), add_residual=add_residual)         # [-1, 32, 8, 8]
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 32, 4, 4]
        
        # fully connected out
        self.linear1 = nn.Linear(4*4*32, 256)
        self.linear2 = nn.Linear(256, 10)
        
        
        
    def conv_forward(self, x):
        """
        Conv part of forward, part of R2R interface.
        """
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        return x
        
        
        
    def fc_forward(self, x):
        """
        Fully connected part of forward, part of R2R interface.
        """
        x = flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
        
        
        
    def out_forward(self, x):
        """
        Output part of forward, part of R2R interface.
        """
        return x
    
    
    
    def forward(self, x):
        """
        nn.Module's forward function
        """
        x = self.conv_forward(x)
        x = self.fc_forward(x)
        return self.out_forward(x)
    
    
    
    def input_shape(self):
        return (1,32,32)



    def hvg(self):
        """
        Build a hvg representing this network
        :return: The HVG for this network
        """
        hvg = HVG(self.input_shape())
        hvg = self.conv_hvg(hvg)
        hvg = self.fc_hvg(hvg)
        return hvg



    def conv_hvg(self, cur_hvg):
        """
        Extend the hvg (containing just the input hvn) with the conv part of the hvg
        :param cur_hvg: The hvg containing just the input hvn
        :return: A hvg representing the entire conv part of this network
        """
        cur_hvg = self.resblock1.conv_hvg(cur_hvg)
        cur_hvg.add_hvn((self.resblock1.r2r.conv2.weight.data.size(0), 16, 16), input_modules=[self.pool1])
        cur_hvg = self.resblock2.conv_hvg(cur_hvg)
        cur_hvg.add_hvn((self.resblock2.r2r.conv2.weight.data.size(0),  8,  8), input_modules=[self.pool2])
        cur_hvg = self.resblock3.conv_hvg(cur_hvg)
        cur_hvg.add_hvn((self.resblock3.r2r.conv2.weight.data.size(0),  4,  4), input_modules=[self.pool3])
        return cur_hvg



    def fc_hvg(self, cur_hvg):
        """
        Adds the linear part of the hvg ontop of the conv part of the hvg
        :param cur_hvg: The hvg which contains the input hvn and the conv hvn
        :return: A complete hvg for the network
        """
        cur_hvg.add_hvn(hv_shape=(self.linear1.out_features,), input_modules=[self.linear1])
        cur_hvg.add_hvn(hv_shape=(self.linear2.out_features,), input_modules=[self.linear2])
        return cur_hvg
        
        