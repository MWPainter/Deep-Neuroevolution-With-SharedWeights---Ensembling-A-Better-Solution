import torch as t
import torch.nn as nn
import torch.nn.functional as F

from resblock import *





all = [Mnist_Resnet]





class Mnist_Resnet(nn.Module):
    """
    A small residual network to be used for mnist/cifar10 tests.
    
    For v2 the only real change to v1 is that we've made all of the layers iterable.
    """
    def __init__(self, identity_initialize=True, noise_ratio=1.0e-8):
        # Superclass initializer
        super(Mnist_Resnet_v2, self).__init__()
        
        # Make the three conv layers, with three max pools
        self.resblock1 = R2R_residual_block_v2(input_channels=1, 
                                               output_channels=8, 
                                               identity_initialize=identity_initialize,
                                               noise_ratio=noise_ratio)                 # [-1, 8, 32, 32]
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 8, 16, 16]     
        self.resblock2 = R2R_residual_block_v2(input_channels=8, 
                                               output_channels=16, 
                                               identity_initialize=identity_initialize,
                                               noise_ratio=noise_ratio)                 # [-1, 16, 16, 16]
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 16, 8, 8]
        self.resblock3 = R2R_residual_block_v2(input_channels=16, 
                                               output_channels=32, 
                                               identity_initialize=identity_initialize,
                                               noise_ratio=noise_ratio)                 # [-1, 32, 8, 8]
        self.pool3 = nn.MaxPool2d(kernel_size=2)                                        # [-1, 32, 4, 4]
        
        # fully connected out
        self.linear1 = nn.Linear(4*4*32, 256)
        self.linear2 = nn.Linear(256, 10)
    
    
    def forward(self, x):
        # convs
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = self.resblock3(x)
        x = self.pool3(x)
        
        # fc
        x = _flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x
    
    
    
    """
    TODO: implement any interfaces needed for widen transforms.
    """
#     def conv_enum(self):
#         """
#         Enumerate through all the conv layers (from network input to output)
#         """
#         return chain(self.resblock1.conv_enum(), 
#                      chain(self.resblock2.conv_enum(), 
#                            self.resblock3.conv_enum()))
        
    
#     def batch_norm_enum(self):
#         """
#         Enumerate through all the batch norm layers (from network input to output)
#         """
#         return chain(self.resblock1.batch_norm_enum(), 
#                      chain(self.resblock2.batch_norm_enum(), 
#                            self.resblock3.batch_norm_enum()))
    
    
#     def fully_connected_enum(self):
#         """
#         Enumerate through all the fully connected layers (from network input to output)
#         """
#         yield self.linear1
#         yield self.linear2
    
    """
    TODO: implement any interfaces needed for deepen transforms.
    """
    
    """
    TODO: wrap the widen and deepen transforms in two convienient helper functions.
    """
        