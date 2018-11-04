import torch as t
import torch.nn as nn
import torch.nn.functional as F

from init_utils import *




# Just export the residual block
all = [Res_Block]





class _R2R_Block(nn.Module):
    """
    Defines a residual block to be used in our mnist and cifar tests.
    This really just aggregates a lot of the work found in the 'tutorial' notebook, and allows us to identity initialize.
    """
    def __init__(self, input_channels, intermediate_channels, output_channels, add_batch_norm=True, zero_initialize=True):
        """
        Initializes each of the layers in the R2R module, and initializes the variables appropriately. That is 
        if zero_initialize == True, then we initialize the network to have a constant zero output.
        
        In the 'shape analysis' above, input_channels, intermediate_channels and output_channels correspond to 
        the variables D, 2*C, O respectively.
        
        :param input_channels: The number of input channels provided to the conv2d layer
        :param intermediate_channels: The number of output channels from the first conv2d layer, and the number 
                of input channels to the 1*1 convolution
        :param output_channels: The number of channels outputted by the whole module
        :param add_max_pool: if we should add a max pool layer at the beginning
        :param add_batch_norm: if we should add a batch norm layer in the middle, before the activation function
        :param zero_initialize: should we initialize the module such that the output is always zero?
        """
        # Superclass initializer
        super(R2R_Block_v1, self).__init__()
        
        self.has_max_pool = add_max_pool
        self.has_batch_norm = add_batch_norm
    
        # Make the layers
        self.opt_max_pool = lambda x: x
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1)
        self.opt_batch_norm = lambda x: x 
        if add_batch_norm:
            self.opt_batch_norm = nn.BatchNorm2d(num_features=intermediate_channels)
        self.activation_function = F.relu
        self.conv2 = nn.Conv2d(intermediate_channels, output_channels, kernel_size=3)

        # If providing a zero initialization, do all of our repeated weight trickery!
        if zero_initialize:
            # Initialize the conv weights as appropriate, using the helpers and adding a small amount of noise
            conv1_filter_shape = (intermediate_channels, input_channels, 3, 3)
            conv1_filter_init = _extend_filter_with_repeated_out_channels(conv1_filter_shape, init_type='He')
            
            self.conv1.weight.data = Parameter(t.Tensor(conv1_filter_init))
            self.conv1.bias.data *= 0.0
            
            conv2_filter_shape = (output_channels, intermediate_channels, 1, 1)
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
        If max pool and batch norm were not specified to be used, then self.opt_max_pool and/or self.opt_batch_norm 
        is a lambda identity function.
        
        :param x: The input tensor
        :return: The output from the module
        """
        x = self.opt_max_pool(x)
        x = self.conv1(x)
        x = self.opt_batch_norm(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        return x
    
    
    
    
    
class Res_Block(nn.Module):
    """
    A small residual block to be used for mnist/cifar10 tests. This is reletively simple, and is just the same 
    as building a small network in PyTorch, using "R2R_Block_v1" as one of the layers.
    
    It consists one set of convolutional layers (not multiple sizes of convolutions like Inception ResNet)
    It has the following architecture:
        Conv2D
        BatchNorm
        ReLU
        Conv2D
        BatchNorm
        ReLU
        _R2R_Block            <- can chose to initialize so that the output is zero (and this block is an identity transform)
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
    def __init__(self, input_channels, intermediate_channels, output_channels, identity_initialize=True, input_shape=None):
        """
        Initialize the filters, optionally making this identity initialized.
        All convolutional filters have the same number of output channels
        """
        # Superclass initializer
        super(Res_Block, self).__init__()
        
        # Check that we gave the correct number of intermediate channels
        if len(intermediate_channels) != 3:
            raise Exception("Need to specify 3 intemediate channels in the resblock")
    
        # The amount of channels to use (forever :( ) in the residual connection
        self.residual_channels = input_channels
        
        # Stuff that we need to remember
        self.input_shape = input_shape
        self.intermediate_channels = intermediate_channels
        self.output_channels = output_channels
        
        # Actual nn.Modules for the network
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.relu = F.relu
        self.conv2 = nn.Conv2d(intermediate_channels[0], intermediate_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        
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
        out = x
        out[:, :self.residual_channels] += res[:, :self.residual_channels]
        
        return out 
    
    
    
    def _get_input_shape(self, input_shape):
        """
        Returns an input shape determined by what is set internally in the resblock, or specified by the input. 
        This sanity checks the input shapes dimensions + that at least one is not None.
        :param input_shape: The input shape to be used if self.input_shape is None
        :return: The input shape
        """
        if self.input_shape is None and input_shape is None:
            raise Exception("Need an input shape somewhere to be able to properly enumerate through layers of the resblock")
         
        shape = self.input_shape if self.input_shape is not None else input_shape
        if len(cur_shape) != 3:
            raise Exception("Input shape needs to be of length 3")
            
        return shape
    
    
    
    def lle(self, input_shape=None):
        """
        Implement the lle (linear layer enum) to iterate through layers for widening.
        Input shape must either not be none here or not be none from before 
        :param input_shape: Shape of the input volume, or, None if it wasn't already specified.
        :return: Iterable over the (in_shape, batch_norm, nn.Module)'s of the resblock
        """
        cur_shape = self._get_input_shape(input_shape)
        
        yield (cur_shape, None, self.conv1)
        cur_shape[0] = self.intermediate_channels[0]
        yield (cur_shape, self.bn1, self.conv2)
        cur_shape[0] = self.intermediate_channels[1]
        yield (cur_shape, self.bn2, self.r2r.conv1)
        cur_shape[0] = self.intermediate_channels[2]
        yield (cur_shape, self.r2r.opt_batch_norm, self.r2r.conv2)
        
        
        
    def extend_hvg(self, cur_hvg, input_node):
        """
        Extends a hidden volume graph 'hvg', from the node 'input_node'
        :param cur_hvg: The HVG object of some larger network (that this resblock is part of)
        :param input_node: The node that this module takes as input
        :return: The hvn for the output from the resblock
        """
        next_shape = self._get_input_shape(input_node.hv_shape)

        # First hidden volume
        next_shape[0] = self.intermediate_channels[0]
        cur_node = cur_hvg.add_hvn(next_shape, input_hvns=[input_node], input_modules=[self.conv1], self.bn1)
        
        # Second hidden volume
        next_shape[0] = self.intermediate_channels[1]
        cur_node = cur_hvg.add_hvn(next_shape, input_hvns=[cur_node], input_modules=[self.conv2], self.bn2)
        
        # Third hidden volume (first of r2r block)
        next_shape[0] = self.intermediate_channels[2]
        cur_node = cur_hvg.add_hvn(next_shape, input_hvns=[cur_node], input_modules=[self.r2r.conv1], 
                                    self.r2r.opt_batch_norm)
        
        # Fourth (output) hidden volume (second of r2r block)
        next_shape[0] = self.output_channels
        out_node = cur_hvg.add_hvn(next_shape, input_hvns=[cur_node], input_modules=[self.r2r.conv2])
        return out_node
        
                   
                   
                   
    def hvg(self):
        raise Exception("hvg() not implemented directly for resblock.")
        
    
    
    
    