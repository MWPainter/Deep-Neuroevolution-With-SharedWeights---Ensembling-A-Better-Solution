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
    def __init__(self, input_channels, intermediate_channels, output_channels, 
                 add_max_pool=False, add_batch_norm=True, zero_initialize=True, noise_ratio=1e-3):
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
        :param noise_ratio: the amount of noise to add, as ratio (of the max init weight in the conv2d kernel) 
                (break symmetry)
        """
        # Superclass initializer
        super(R2R_Block_v1, self).__init__()
        
        self.has_max_pool = add_max_pool
        self.has_batch_norm = add_batch_norm
    
        # Make the layers
        self.opt_max_pool = lambda x: x
        if add_max_pool:
            self.opt_max_pool = nn.MaxPool2d(kernel_size=2)
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
    def __init__(self, input_channels, output_channels, identity_initialize=True, noise_ratio=0.0):
        """
        Initialize the filters, optionally making this identity initialized.
        All convolutional filters have the same number of output channels
        """
        # Superclass initializer
        super(Res_Block, self).__init__()
    
        self.residual_channels = input_channels
        self.noise_ratio = noise_ratio
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=output_channels)
        self.relu = F.relu
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=output_channels)
        
        self.r2r = _R2R_Block(output_channels, output_channels, output_channels, 
                              zero_initialize=identity_initialize, noise_ratio=noise_ratio)
        
        
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
    
    
    """
    TODO: implement any interfaces needed for widen transforms.
    """
#     def conv_enum(self):
#         """
#         Enumerate through all the convolutional layers (from network input to output)
#         """
#         yield self.conv1
#         yield self.conv2
#         yield self.r2r.conv1
#         yield self.r2r.conv2
        
    
#     def batch_norm_enum(self):
#         """
#         Enumerate through all the batch norm layers (from network input to output)
#         """
#         yield self.bn1
#         yield self.bn2
#         if self.r2r.has_batch_norm:
#             yield self.r2r.opt_batch_norm
    
    """
    Note: We don't implement an interface for deepening transforms, because we don't need to deepen out residual blocks.
    """
    
    
    