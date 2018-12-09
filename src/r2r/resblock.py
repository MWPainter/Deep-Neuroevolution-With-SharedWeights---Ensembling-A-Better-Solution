import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from r2r.init_utils import *
from r2r.residual_connection import Residual_Connection




# Just export the residual block
__all__ = ['Res_Block']





class _R2R_Block(nn.Module):
    """
    Defines a small convolutional block, with 2 layers, which can be initialized such that .
    That is, if this block is represented by the function f, then for all x, we can set f(x)=0.
    """
    def __init__(self, input_channels, intermediate_channels, output_channels, add_batch_norm=True,
                 zero_initialize=True):
        """
        Creates a nn.Module with 2 convolutional layers. If 'zero_initialize' is true, then we initialize the
        convolutions such that the output is a constant zero.
        
        :param input_channels: The number of input channels provided to the conv2d layer
        :param intermediate_channels: The number of output channels from the first conv2d layer, and the number 
                of input channels to the 1*1 convolution
        :param output_channels: The number of channels outputted by the whole module
        :param add_max_pool: if we should add a max pool layer at the beginning
        :param add_batch_norm: if we should add a batch norm layer in the middle, before the activation function
        :param zero_initialize: should we initialize the module such that the output is always zero?
        """
        # Superclass initializer
        super(_R2R_Block, self).__init__()

        self.has_batch_norm = add_batch_norm

        # Make the layers
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1)
        self.opt_batch_norm = lambda x: x 
        if add_batch_norm:
            self.opt_batch_norm = nn.BatchNorm2d(num_features=intermediate_channels)
        self.activation_function = F.relu
        self.conv2 = nn.Conv2d(intermediate_channels, output_channels, kernel_size=3, padding=1)

        # To provide a zero initialization, initialize weights symmetrically such that the function is identically zero.
        if zero_initialize:
            # Initialize the conv weights as appropriate, using the helpers and adding a small amount of noise
            conv1_filter_shape = (intermediate_channels, input_channels, 3, 3)
            conv1_filter_init = _extend_filter_with_repeated_out_channels(conv1_filter_shape, init_type='He')
            
            self.conv1.weight.data = Parameter(t.Tensor(conv1_filter_init))
            self.conv1.bias.data *= 0.0
            
            conv2_filter_shape = (output_channels, intermediate_channels, 3, 3)
            conv2_filter_init = _extend_filter_with_repeated_in_channels(conv2_filter_shape, init_type='He', 
                                                                       alpha=-1.0)
            
            self.conv2.weight.data = Parameter(t.Tensor(conv2_filter_init))
            self.conv2.bias.data *= 0.0
            
            # Initialize the batch norm variables so that the scale is one and the mean is zero
            if add_batch_norm:
                self.opt_batch_norm.weight.data = Parameter(t.Tensor(t.ones(intermediate_channels)))
                self.opt_batch_norm.bias.data = Parameter(t.Tensor(t.zeros(intermediate_channels)))
                
            
    def forward(self, x):
        """
        Forward pass of the module.
        If we chose not to add batch_norm, then self.opt_batch_norm is an identity lambda function.
        
        :param x: The input tensor
        :return: The output from the module
        """
        x = self.conv1(x)
        x = self.opt_batch_norm(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        return x
    
    
    
    
    
class Res_Block(nn.Module):
    """
    A small residual block to be used for mnist/cifar10 tests.
    
    It consists one set of convolutional layers (not multiple sizes of convolutions like Inception ResNet)
    It has the following architecture:
        Conv2D
        BatchNorm
        ReLU
        Conv2D
        BatchNorm
        ReLU
        _R2R_Block              <- can chose to initialize so that the output is zero (making this entire resblock is an identity transform)
            Conv2D
            BatchNorm
            ReLU
            Conv2D
            (implicit linear activation)

        + residual connection

    To deal with dimensionality issues, we use zero padding for the residual connection. We also use a
    bit of masking in the zero padding, so that the residual connection is always the same shape.
    
    It would be good to fix this limitation of not being able to increase the residual connection size.
    """
    def __init__(self, input_channels, intermediate_channels, output_channels, identity_initialize=True, 
                 input_spatial_shape=None, input_volume_slices_indices=None, add_residual=True):
        """
        Initialize the conv layers and so on, optionally making this identity initialized.

        :param input_channels: The number of channels input to the res block
        :param intermediate_channels: A list of 3 numbers, specifying the intermedaite numbers of channels for the 3
            intermediate volumes.
        :param output_channels: The number of channels for the volume output by the resblock.
        :param identity_initialize: If the resblock should be initialized such that it represents an identity function.
        :param input_spatial_shape: The spatial dimensions of the input shape.
        :param input_volume_slices_indices: The slices of the input volume to the residual block (i.e. if the input to
                this block is the concatenation of two volumes, with 10 channels and 20 channels respectively, then we
                should have input_volume_slices_indices = [0,10,30]). If None, then we will assume
                    input_volume_slices_indices=[0,input_channels], which means that the input is from a single volume.
        """
        # Superclass initializer
        super(Res_Block, self).__init__()

        self.add_residual = add_residual
        
        # Check that we gave the correct number of intermediate channels
        if len(intermediate_channels) != 3:
            raise Exception("Need to specify 3 intemediate channels in the resblock")

        # Make the residual connection object (using the input_volume_slices_inddices)
        if input_volume_slices_indices is None:
            input_volume_slices_indices = [0, input_channels]
        self.residual_connection = Residual_Connection(input_volume_slices_indices)

        # Stuff that we need to remember
        self.input_spatial_shape = input_spatial_shape
        self.intermediate_channels = intermediate_channels
        self.output_channels = output_channels
        
        # Actual nn.Modules for the network
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=intermediate_channels[0])
        self.relu = F.relu
        self.conv2 = nn.Conv2d(intermediate_channels[0], intermediate_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=intermediate_channels[1])
        
        self.r2r = _R2R_Block(intermediate_channels[1], intermediate_channels[2], output_channels,
                              zero_initialize=identity_initialize)
        


    def forward(self, x):
        """
        Forward pass through this residual block
        
        :param x: the input
        :return: THe output of applying this residual block to the input
        """
        # Forward pass through residual part of the network
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.r2r(x)
        
        # add residual connection (implicit zero padding)
        if self.add_residual:
            out = self.residual_connection(x, res)
        else:
            out = x
        
        return out
    
    

    def conv_lle(self):
        """
        Conv part of the 'lle' function.
        Part of the R2DeeperRBlock interface.

        :return: Iterable over the (in_shape, batch_norm, nn.Module)'s of the resblock
        """
        height, width = self.input_spatial_shape

        # 1st (not 0th) dimension is the number of in channels of a conv, so (in_channels, height, width) is input shape
        yield ((self.conv1.weight.data.size(1), height, width), None, self.conv1, self.residual_connection)
        yield ((self.conv2.weight.data.size(1), height, width), self.bn1, self.conv2, None)
        yield ((self.r2r.conv1.weight.data.size(1), height, width), self.bn2, self.r2r.conv1, None)
        yield ((self.r2r.conv2.weight.data.size(1), height, width), self.r2r.opt_batch_norm, self.r2r.conv2, None)




    def lle(self):
        """
        Implement the lle (linear layer enum) to iterate through layers for widening.

        :return: Iterable over the (in_shape, batch_norm, nn.Module)'s of the resblock
        """
        return self.conv_lle()
        
        
        
    def conv_hvg(self, cur_hvg):
        """
        Extends a hidden volume graph 'hvg'.
        Part of the R2DeeperRBlock interface.

        :param cur_hvg: The HVG object of some larger network (that this resblock is part of)
        :return: The hvn for the output from the resblock
        """
        # Raise an error if the cur_hvg has more than one output hidden volume node. It must necessarily be one
        # to be able to apply a residual connection.
        output_nodes = cur_hvg.get_output_nodes()
        if len(output_nodes) > 1:
            raise Exception("Input to residual block when making HVG was multiple volumes.")

        # Add the residual connection object to the current hvg output node
        output_node = output_nodes[0]
        if self.add_residual:
            output_node.residual_connection = self.residual_connection

        # First hidden 
        cur_hvg.add_hvn((self.conv1.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]),
                        input_modules=[self.conv1], batch_norm=self.bn1)
        
        # Second hidden volume
        cur_hvg.add_hvn((self.conv2.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]), 
                        input_modules=[self.conv2], batch_norm=self.bn2)
        
        # Third hidden volume (first of r2r block)
        cur_hvg.add_hvn((self.r2r.conv1.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]), 
                        input_modules=[self.r2r.conv1], batch_norm=self.r2r.opt_batch_norm)
        
        # Fourth (output) hidden volume (second of r2r block)
        cur_hvg.add_hvn((self.r2r.conv2.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]), 
                         input_modules=[self.r2r.conv2])
        return cur_hvg
        
                   
                   
                   
    def hvg(self):
        raise Exception("hvg() not implemented directly for resblock.")
        
    
    
    
    