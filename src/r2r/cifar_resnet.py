import torch as t
import torch.nn as nn
import torch.nn.functional as F

from r2r.resblock import *
from utils import flatten

from itertools import chain





__all__ = ['Cifar_Resnet']





class Cifar_Resnet(nn.Module):
    """
    A small residual network to be used for cifar tests.

    Implements the R2R interface.
    """

    def __init__(self, identity_initialize=True, add_residual=True):
        # Superclass initializer
        super(Cifar_Resnet, self).__init__()

        # Make the three conv layers, with three max pools
        self.resblock1 = Res_Block(input_channels=3, intermediate_channels=[16,16,16],
                                   output_channels=16, identity_initialize=identity_initialize,
                                   input_spatial_shape=(32,32), add_residual=add_residual)                                  # [-1, 8, 32, 32]
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 8, 16, 16]
        self.resblock2 = Res_Block(input_channels=16, intermediate_channels=[32,32,32],
                                   output_channels=32, identity_initialize=identity_initialize,
                                   input_spatial_shape=(16,16), add_residual=add_residual)                                  # [-1, 16, 16, 16]
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 16, 8, 8]
        self.resblock3 = Res_Block(input_channels=32, intermediate_channels=[64,64,64],
                                   output_channels=64, identity_initialize=identity_initialize,
                                   input_spatial_shape=(8,8))                                    # [-1, 32, 8, 8]
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                                 # [-1, 32, 4, 4]

        # fully connected out
        self.linear1 = nn.Linear(4*4*64, 256)
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

    def lle_or_hvg(self):
        """
        Return if we're using the lle or hvg interface.

        Part of the R2R interface.
        """
        return "lle"

    def input_shape(self):
        return (3, 32, 32)

    def conv_lle(self):
        """
        Enumerate through all the conv layers (from network input to output), returning (shape, batch_norm, nn.Module) tuples

        Part of the R2R interface.
        """
        return chain(self.resblock1.conv_lle(),
                     chain(self.resblock2.conv_lle(),
                           self.resblock3.conv_lle()))

    def fc_lle(self):
        """
        Enumerate through all the fc layers (from network input to output), returning (shape, batch_norm, nn.Module) tuples

        Part of the R2R interface.
        """
        yield ((self.linear1.in_features,), None, self.linear1, None)
        yield ((self.linear1.out_features,), None, self.linear2, None)
        yield ((self.linear2.out_features,), None, None, None)

    def lle(self):
        """
        Implement the linear layer enumeration (lle), to be able to interface with the wider transforms.

        Part of the R2R interface.
        """
        return chain(self.conv_lle(), self.fc_lle())


        