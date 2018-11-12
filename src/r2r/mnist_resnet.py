import torch as t
import torch.nn as nn
import torch.nn.functional as F

from r2r.resblock import *
from utils import flatten

from itertools import chain





__all__ = ['Mnist_Resnet']





class Mnist_Resnet(nn.Module):
    """
    A small residual network to be used for mnist/cifar10 tests.
    
    For v2 the only real change to v1 is that we've made all of the layers iterable.
    """
    def __init__(self, identity_initialize=True):
        # Superclass initializer
        super(Mnist_Resnet, self).__init__()
        
        # Make the three conv layers, with three max pools
        self.resblock1 = Res_Block(input_channels=1, intermediate_channels=[8,8,8], 
                                   output_channels=8, identity_initialize=identity_initialize,
                                   input_spatial_shape=(32,32))                                  # [-1, 8, 32, 32]
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 8, 16, 16]    
        self.resblock2 = Res_Block(input_channels=8, intermediate_channels=[16,16,16],
                                   output_channels=16, identity_initialize=identity_initialize,
                                   input_spatial_shape=(16,16))                                  # [-1, 16, 16, 16]
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 16, 8, 8]
        self.resblock3 = Res_Block(input_channels=16, intermediate_channels=[32,32,32],
                                   output_channels=32, identity_initialize=identity_initialize,
                                   input_spatial_shape=(8,8))                                    # [-1, 32, 8, 8]
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 32, 4, 4]
        
        # fully connected out
        self.linear1 = nn.Linear(4*4*32, 256)
        self.linear2 = nn.Linear(256, 10)
        
        
        
    def conv_forward(self, x):
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        return x
        
        
        
    def fc_forward(self, x):
        x = flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
        
        
        
    def out_forward(self, x):
        return x
    
    
    
    def forward(self, x):
        x = self.conv_forward(x)
        x = self.fc_forward(x)
        return self.out_forward(x)
    
    
    
    def lle_or_hvg(self):
        """
        Return if we're using the lle or hvg interface
        """
        return "lle"
    
    
    
    def input_shape(self):
        return (1,32,32)
    
    
    
    def conv_lle(self):
        """
        Enumerate through all the conv layers (from network input to output), returning (shape, batch_norm, nn.Module) tuples
        """
        return chain(self.resblock1.conv_lle(), 
                     chain(self.resblock2.conv_lle(), 
                           self.resblock3.conv_lle()))
    
    
    
    def fc_lle(self):
        """
        Enumerate through all the fc layers (from network input to output), returning (shape, batch_norm, nn.Module) tuples
        """
        yield ((self.linear1.in_features,), None, self.linear1)
        yield ((self.linear1.out_features,), None, self.linear2)
        yield ((self.linear2.out_features,), None, None)
        
        
        
    def lle(self):
        """
        Implement the linear layer enumeration (lle), to be able to interface with the wider transforms.
        """
        return chain(self.conv_lle(), self.fc_lle())
        
        