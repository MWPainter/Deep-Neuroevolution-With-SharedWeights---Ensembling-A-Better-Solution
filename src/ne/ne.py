"""
THIS FILE CURRENTLY CONTAINS HISTORIC CODE THAT MAY NOT RUN. (I can't remember if it was self contained)

This was the original neuroevolution code at the close of the project.

We need to clean this up, and haven't made any attempt to clean it up yet.
"""



all = []














from __future__ import print_function
import random
import time
import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import copy

import os
import sys
dataset_module_path = os.path.abspath(os.path.join('..'))
if dataset_module_path not in sys.path:
    sys.path.append(dataset_module_path)
from dataset import MnistDataset
from dataset import CifarDataset
from dataset import DatasetCudaWrapper

import pprint
import pickle

from flops_utils import *



def save(dic, dic_name):
    with open("./results/%s.dat" % dic_name, "wb") as f:
        pickle.dump(dic, f)







def _zero_pad_1d(old_val, new_params):
    """
    Zero pads an old pytorch tensor to match the new number of outputs
    
    :param old_val the old pytorch tensor to zero pad
    :param new_params: the number of new params needed
    :return: a new, padded tensor
    """
    old_len = old_val.size()[0]
    canvas = t.zeros((old_len+new_params,))
    canvas[:old_len] = old_val
    return canvas



def _one_pad_1d(old_val, new_params):
    """
    One pads an old pytorch tensor to match the new number of outputs
    
    :param old_val the old pytorch tensor to one pad
    :param new_params: the number of new params needed
    :return: a new, padded tensor
    """
    old_len = old_val.size()[0]
    canvas = t.ones((old_len+new_params,))
    canvas[:old_len] = old_val
    return canvas





def _conv_xavier_initialize(filter_shape, override_input_channels=None, override_output_channels=None):
    """
    Initialize a convolutional filter, with shape 'filter_shape', according to "Xavier initialization".
    The weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of 
    sqrt(2/(n_in + n_out)).
    
    This is the initialization of choice for layers with non ReLU activations.
    
    The filter shape should be [output_channels, input_channels, width, height]. So here, n_in = input_channels 
    and n_out = width * height * output_channels.
    
    :param filter_shape: THe shape of the filter that we want to produce an initialization for
    :param override_output_channels: Override for the number of input filters in the filter_shape (optional)
    :param override_output_channels: Override for the number of output filters in the filter_shape (optional)
    :return: A numpy array, of shape 'filter_shape', randomly initialized according to He initialization.
    """
    out_channels, in_channels, width, height = filter_shape
    if override_input_channels is not None:
        in_channels = override_input_channels
    if override_output_channels is not None:
        out_channels = override_output_channels
    filter_shape = (width, height, in_channels, out_channels)    
    
    scale = np.sqrt(2.0 / (in_channels + width*height*out_chanels))
    return scale * np.random.randn(*filter_shape).astype(np.float32) 
    
    

    
    
def _conv_he_initialize(filter_shape, override_input_channels=None):
    """
    Initialize a convolutional filter, with shape 'filter_shape', according to "He initialization".
    Each weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of 
    sqrt(2/n_in).
    
    This is the initialization of choice for layers with ReLU activations.
    
    The filter shape should be [output_channels, input_channels, width, height]. So here, n_in = input_channels.
    
    As the initization only depends on the number of inputs (the number of input channels), unlike Xavier 
    initialization, we don't need to be able to override the number of output_channels.
    
    :param filter_shape: THe shape of the filter that we want to produce an initialization 
    :param override_output_channels: Override for the number of input filters in the filter_shape (optional)
    :return: A numpy array, of shape 'filter_shape', randomly initialized according to He initialization.
    """
    in_channels = filter_shape[1]
    if override_input_channels is not None:
        in_channels = override_input_channels
    scale = np.sqrt(2.0 / in_channels)
    return scale * np.random.randn(*filter_shape).astype(np.float32)





def _conv_stddev_initialize(filter_shape, stddev):
    """
    Randomly initializes (normal distribution) numpy array, with shape 'filter_shape', 
    zero mean and stddev of 'stddev'.
    
    :param filter_shape: The shape of the np array to initialize (for a new kernel)
    :param stddev: The stddev to initialized the new kernel with
    :return: The new kernel, initialized with an appropriate stddev, as a np array
    """
    return stddev * np.random.randn(*filter_shape).astype(np.float32)





def _widen_kernel_with_repeated_in_repeated_output(old_kernel, old_bias, extra_in_channels=0, 
                                                   extra_out_channels=0, repeated_in=True, 
                                                   repeated_out=True, init_type='stddev_match',
                                                   scale=0.5):
    """
    Helper function to widen a kernel. We pass in the old kernel, the number of channels 
    extended in the convolution before it (zero if the convolution before it hasn't been 
    widened), and the number of new channels for the output.
    
    If the previous convolution has been widened, the new output channels will be repeated.
    We want to negate those inputs in this convolution, so the new params in the input 
    don't change this convolutions output at all. Performing this operation is controlled 
    by the 'repeated_in' boolean. Set this to false if we *don't* want to perform this operaiton.
    
    If we have new output filters, repeate the output if the 'repeated_out' boolean is true. 
    This is so that we can negate it later, to provide a zero initialization.
    
    Let the old shape be [Ou,I,W,H] and the new shape be [Ou2,I2,W,H]. So Ou2 is the new number 
    of output channels and Ou is the old number of output channels, when we initialize as follows:
    
    Letting hOu = (Ou2-Ou)/2
    and hI = (I2-I)/2
    and r1, r2, r3 are appropriately randomly initialized arrays,
    then 'repeated_in = True' and 'repeated_out = True' will initialize the filters as follows:
    
    
        In \ Out |      0:Ou      |    Ou:Ou+hOu    |   Ou+hOu:Ou2    
        -------------------------------------------------------------
             0:I |  old[0:Ou,0:I] |  r1[0:hOu,0:I]  |  r1[0:hOu,0:I]  
        -------------------------------------------------------------       
          I:I+hI |  r2[0:Ou,0:hI] |  r3[0:hOu,0:hI] |  r3[0:hOu,0:hI] 
        -------------------------------------------------------------
         I+hI:I2 | -r2[0:Ou,0:hI] | -r3[0:hOu,0:hI] | -r3[0:hOu,0:hI]   
         
         
    Note, if we change either repeated_input or repeated_output, then, we no longer have a function preserving
    transfom. And we ignore the above grid. We provide this as an option so that we can implment "random_padding"
    widen transforms (not function preserving).
    
    :param old_kernel: the old kernel of the convolution layer (of shape (Ou,I,W,H))
    :param old_bias: the old bias of the convolution layer (of shape (Ou,))
    :param extra_in_channels: the number of new channels being input to this layer (Ou2-Ou)
    :param extra_out_channels: the number of new channels being output from this layer (I2-I)
    :param repeated_in: if we want to handle repeated input (of NEW input channels)
    :param repeated_out: if we want to make the NEW output channels repeated 
    :param init_type: The type of initialization to use for the kernel
    :param scale: A constant factor to multiply the output by (to scale the stddev appropriately)
    :return: A new, larger filter and bias. (Initialized appropriately for a function preserving transform)
    """
    if (repeated_in and extra_in_channels % 2 != 0) or (repeated_out and extra_out_channels % 2 != 0):
        err = ("Tried to widen a convolutional filter (function preserving), "
               "with an odd number of new filters")
        raise Exception(err)
        
    # compute values related to kernel size
    Ou, I, W, H = old_kernel.size()
    hOu = extra_out_channels // 2
    hI = extra_in_channels // 2
    total_new_in_channels = I + extra_in_channels
    total_new_out_channels = Ou + extra_out_channels
    
    # init function
    init = None
    if init_type == "He":
        init = lambda shape: t.tensor(_conv_he_initialize(shape, total_new_in_channels, total_new_out_channels))
    elif init_type == "Xavier":
        init = lambda shape: t.tensor(_conv_xavier_initialize(shape, total_new_in_channels, total_new_out_channels))
    elif init_type == "stddev_match":
        weight_stddev = t.std(old_kernel).cpu().numpy()
        init = lambda shape: t.tensor(_conv_stddev_initialize(shape, weight_stddev))
    else:
        err = "Invalid init_type used when trying to widen the kernel"
        raise Exception(err)
    
    # compute r1, r2, r3 as above.
    if hOu > 0:
        r1 = init((hOu,  I, W, H)) 
    if hI > 0:
        r2 = init(( Ou, hI, W, H))
    if hOu > 0 and hI > 0:
        r3 = init((hOu, hI, W, H))
    
    # make a canvas and fill it appropriately
    # ignore repetitions appropriately if either negate_repeated_new_input, repeat_new_output is false
    canvas = t.zeros((total_new_out_channels, total_new_in_channels, W, H))
    
    # top left four squares
    canvas[:Ou, :I] = old_kernel
    if hOu > 0:
        canvas[Ou:Ou+hOu, :I] = r1
    if hI > 0:
        canvas[:Ou, I:I+hI] = r2
    if hOu > 0 and hI > 0:
        canvas[Ou:Ou+hOu, I:I+hI] = r3
    
    # bottom left two squares
    if hI > 0:
         canvas[:Ou, I+hI:I+2*hI] = -r2 if repeated_in else init((Ou, hI, W, H))
    if hOu > 0 and hI > 0:
        canvas[Ou:Ou+hOu, I+hI:I+2*hI] = -r3 if repeated_in else init((hOu, hI, W, H))
        
    # right three squares    
    if hOu > 0:
        canvas[Ou+hOu:Ou+2*hOu] = canvas[Ou:Ou+hOu] if repeated_out else init((hOu, I+2*hI, W, H))
        
    # scale
    canvas *= scale
        
    # Bias just needs to be zero padded appropriately
    return canvas, _zero_pad_1d(old_bias, extra_out_channels)





class R2R_conv(nn.Module):
    """
    Wrapper around nn.conv2d, adding the logic to appropriately initialize the kernel for a 
    function preserving transform, and to provide a widening operation.
    
    TODO: Try to actually subclass nn.Conv2d, rather than having a full on wrapper
    
    TODO: Actually implement and utilize "widen", maybe?
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 adapt_for_repeated_in_channels=False, repeat_new_output_channels=True, noise_ratio=0.0, init_type="He"):
        """
        Initializer. Takes two additional arguments above the normal Conv2D arguments, specifying if we 
        need to adapt the weights of the transform when initializing/widening. Takes another additional 
        argument for the type of initialization to use.
        
        :param in_channels: in_channels for conv2d.
        :param out_channels: out_channels for conv2d.
        :param kernel_size: kernel_size for conv2d.
        :param stride: stride for conv2d.
        :param padding: padding for conv2d.
        :param dilation: dilation for conv2d.
        :param groups: groups for conv2d.
        :param bias: bias for conv2d.
        :param adapt_for_repeated_in_channels: Does the input consist of repeated output, that we need to negate?
        :param repeat_new_output_channels: Should we duplicate weights in the output channels?
        :param noise_ratio: The amount of noise to add for symmetry breaking in widening.
        :param init_type: the type of init to use. 
        """
        # Superclass initializer
        super(R2R_conv, self).__init__()
        
        # make our conv 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # compute/remember params
        self.adapt_for_repeated_in_channels = adapt_for_repeated_in_channels
        self.repeat_new_output_channels = repeat_new_output_channels
        self.noise_ratio = noise_ratio
        
        self.filter_shape = None
        if type(kernel_size) is int:
            self.filter_shape = (out_channels, in_channels, kernel_size, kernel_size)
        elif len((kernel_size)) == 2:
            self.filter_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        else:
            raise Exception("Invalid kernel size input.")
            
        # init function
        init = None
        if init_type == "He":
            init = _conv_he_initialize
        elif init_type == "Xavier":
            init = _conv_xavier_initialize
        else:
            raise Exception("Invalid init_type specified. Must be 'He' or 'Xavier'.")
            
        # Init the weight and bias how WE want it
        self.conv.weight.data = Parameter(t.tensor(init(self.filter_shape)))
        self.conv.bias.data *= 0
        
        # Cancel out duplicated channels in if needed
        if adapt_for_repeated_in_channels:
            if in_channels % 2 != 0:
                raise Exception("Can't cancel channels being input if there isn't an even number of them.")
                
            half_in_channels = in_channels // 2
            self.conv.weight.data[:,:half_in_channels] = -self.conv.weight.data[:,half_in_channels:]
            
        # Duplicate the output channels if needed
        if repeat_new_output_channels:
            if out_channels % 2 != 0:
                raise Exception("Can't duplicate output channels if there isn't an even number of them.")
                
            half_out_channels = out_channels // 2
            self.conv.weight.data[:half_out_channels] = self.conv.weight.data[half_out_channels:]
        
        # Add noise
        noise_stddev = noise_ratio * float(t.max(t.abs(self.conv.weight.data)))
        self.conv.weight.data += t.tensor(_conv_stddev_initialize(self.filter_shape, stddev=noise_stddev))

    def forward(self, x):
        return self.conv(x)
            
#     def widen(self, new_in_channels, new_out_channels, init_type="stddev_match"):
#         """
#         Widen this convolution to produce a new one. Involves zero padding the bias and extending the 
#         number of input and output channels appropriately. And really just involves handing off to the 
#         appropriate _widen_kernel_with_repeated_in_repeated_output function
        
#         :param new_in_channels: The number of new input channels
#         :param new_out_channels: The number of new output channels
#         :param init_type: The init to use for the new params
#         """
#         widen = _widen_kernel_with_repeated_in_repeated_output
#         new_kernel, new_bias = widen(self.weight.data, self.bias.data, new_in_channels, new_out_channels,
#                                      self.adapt_for_repeated_in_channels, self.repeat_new_output_channels, 
#                                      init_type)
        
#         noise_stddev = self.noise_ratio * t.max(t.abs(new_kernel)) 
#         new_kernel += _conv_stddev_initialize(filter_shape, stddev=noise_stddev)
        
#         self.weight = Parameter(new_kernel)
#         self.bias = Parameter(new_bias)
        
        
        
        
class R2R_block(nn.Module):
    """
    Basically a c&p from the deriving Resnet2Resnet
    TODO: actually write thie description properly
    TODO: remove the need to have an odd sized kernel
    TODO: maybe add more options, allowing padding to be specified etc?
    """
    def __init__(self, input_channels, intermediate_channels, output_channels, kernel_size, add_batch_norm=True, 
                 zero_initialize=True, noise_ratio=0.0):
        """
        Initializes each of the layers in the R2R module, and initializes the variables appropriately 
        
        :param input_channels: THe number of input channels provided to the conv2d layer
        :param intermediate_channels: The number of output channels from the first conv2d layer, and the number input to the 
                1*1 convolution
        :param output_channels: The number of channels output by the whole module
        :param add_max_pool: if we should add a max pool layer at the beginning
        :param add_batch_norm: if we should add a batch norm layer in the middle, before the activation function
        :param zero_initialize: should we initialize the module such that the output is always zero?
        :param noise_ratio: the amount of noise to add, as ratio (of the max init weight in the conv2d kernel) (break symmetry)
        """
        # Superclass initializer
        super(R2R_block, self).__init__()
        
        # only allow odd kernel sizes
        if kernel_size % 2 == 0:
            raise Exception("R2R block needs a initialized with an odd kernel size (for now)")
        padding = (kernel_size - 1) // 2
    
        # Make conv layers (n.b. 'zero_initialize' appropriately used)
        self.conv = R2R_conv(input_channels, intermediate_channels, kernel_size=kernel_size, padding=padding,
                             adapt_for_repeated_in_channels=False, repeat_new_output_channels=zero_initialize,
                             noise_ratio=noise_ratio)
        self.activation_function = F.relu
        self.reduction_conv = R2R_conv(intermediate_channels, output_channels, kernel_size=1, padding=0,
                                       adapt_for_repeated_in_channels=zero_initialize, repeat_new_output_channels=False,
                                       noise_ratio=noise_ratio)
        
        # make batch norm
        self.batch_norm = None
        if add_batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=intermediate_channels)
        if zero_initialize:
            self.batch_norm.weight.data = Parameter(t.Tensor(t.ones(intermediate_channels)))
            self.batch_norm.bias.data = Parameter(t.Tensor(t.zeros(intermediate_channels)))
            
            
        # store a activation function and anything else needed
        self.activation_finction = F.relu
        self.zero_initialized = zero_initialize
        self.out_channels = output_channels
        
    
    def get_out_channels(self):
        return self.out_channels
            
            
            
    def forward(self, x):
        """
        Forward pass of the module.
        
        :param x: The input tensor
        :return: The output from the module
        """
        x = self.conv(x)
        if self.batch_norm is not None: 
            x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.reduction_conv(x)
        return x







class R2R_filter_bank(object):
    """
    TODO: we're just adding all of the filter right now, really, should be concatenating... (difficult with shaping info)
    TODO: add reference counting and deleting models from datastructs
    TODO: deal with having more layers per residual connection?
    """
    def __init__(self, fc_hidden_size = 128, cuda=True):
        """
        Initializes our layers data struct.
        self.layers: layer_id -> (block_id -> r2r_block)
        """        
        self.layers = {}
        self.next_layer_id = 0
        
        self.initialized = False
        
        # because we're making blocks here, we need to put them on the GPU now
        self.cuda = cuda
        
        # setup the fc layers
        self.fc_input_size = 64 * 4 * 4
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = 10
        
        self.fc1 = nn.Linear(self.fc_input_size, self.fc_hidden_size)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.num_classes)
        
        if self.cuda:
            self.fc1 = self.fc1.cuda()
            self.fc2 = self.fc2.cuda()
        
        self.fcs = [self.fc1, self.fc2]
        
        # construct base dna to be used
        self.base_dna = self._construct_initial_dna()
        
        
    def gen_layer(self, in_channels, out_channels):
        """
        Generates a new layer in the filter bank. 
        Returns id of that new layer.
        """
        layer_id = self.next_layer_id
        self.next_layer_id += 1
        self.layers[layer_id] = {'layer_id':layer_id,
                                 'maxpool': False,
                                 'in':in_channels,
                                 'out':out_channels,
                                 'blocks':{},
                                 'next_block_id':0}
        return layer_id
    
    def gen_maxpool_layer(self, channels):
        """
        Adds a new maxpool layer.
        Returns the id of that layer.
        """
        layer_id = self.next_layer_id
        self.next_layer_id += 1
        maxpool = nn.MaxPool2d(kernel_size=2)
        self.layers[layer_id] = {'layer_id': layer_id,
                                 'maxpool': True,
                                 'in':channels,
                                 'out':channels,
                                 'blocks':{0: maxpool}}
        return layer_id
        
    def widen_layer(self, layer_id, kernel_size, intermediate_channels):
        """
        Adds a new R2R_conv in a layer
        Intermediate_channels = within R2R block
        """
        if self.layers[layer_id]['maxpool']:
            raise Exception("Tried to widen a max pool layer")
        
        layer = self.layers[layer_id]
        in_channels = layer['in']
        out_channels = layer['out']
        block = R2R_block(in_channels, intermediate_channels, out_channels, kernel_size)
        if self.cuda:
            block = block.cuda()
        block_id = layer['next_block_id']
        layer['blocks'][block_id] = block
        layer['next_block_id'] = block_id + 1
        return block_id
        
    def get_layer(self, layer_id):
        """
        Lookup layer struct
        """
        return self.layers[layer_id]
    
    def get_block(self, layer_id, block_id):
        """
        Lookup r2r block
        """
        return self.layers[layer_id]['blocks'][block_id]
    
    def get_fully_connected_layers(self):
        """
        Getter for the fully connected layers. 
        Also indicates the dimension of the input to the fully connected layers.
        """
        return self.fc_input_size, self.fcs
    
    def get_initial_dna(self):
        """
        getter for the initial, base, dna
        """
        return self.base_dna
        
    def _construct_initial_dna(self):
        """
        Initializes the filter bank with a small number of filters, and constructs the initial dna
        """
        if self.initialized:
            raise Exception("Duplicate call to construct_initial_dna().")
        self.initialized = False
        
        return [self._construct_r2r_layer(3, 8, 16),              # (16, 32, 32)
                (self.gen_maxpool_layer(16), [0]),                # (16, 16, 16)
                self._construct_r2r_layer(16, 16, 32),            # (32, 16, 16)
                (self.gen_maxpool_layer(32), [0]),                # (32,  8,  8)
                self._construct_r2r_layer(32, 32, 64),            # (64,  8,  8) 
                (self.gen_maxpool_layer(64), [0])]                # (64,  4,  4)
        
    def _construct_r2r_layer(self, in_channels, intermediate_channels, out_channels):
        """
        Internal helper for construct_initial_dna
        """
        layer_id = self.gen_layer(in_channels, out_channels)
        block_id = self.widen_layer(layer_id, 3, intermediate_channels)
        return (layer_id, [block_id])

    def print(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.layers)



class R2R_Cifar_Resnet(nn.Module):
    """
    TODO: clean up
    TODO: make all neat and shit
    TODO: document properly (make lightweight objects, R2R_filter_bank_layer, etc, to encapsulate all of the info for it
                            rather than putting everything into maps...)
    TODO: properly deal with the hack that we did (just find some way to explicitly call the handles to add params to module)
    """
    def __init__(self, filter_bank, dna):
        super(R2R_Cifar_Resnet, self).__init__()
        
        self.dna = dna
        self.construct_network(filter_bank, dna)
        self.filter_bank = filter_bank
        
    def construct_network(self, filter_bank, dna):
        # organize and store refs for the r2r blocks, layer by layer
        # hackily, we create a parameter list, storing all of the parameters
        # otherwise it's not added to pytorch's internal param list
        self._param_list = nn.ParameterList()
        self.blocks = []
        for layer_id, block_ids in dna:
            block_set = []
            for block_id in block_ids:
                r2r_block = filter_bank.get_block(layer_id, block_id)
                block_set.append(r2r_block)
                self._param_list.extend(r2r_block.parameters())
            self.blocks.append((layer_id, block_set))
            
        # store refs for the fc layers
        self.fc_dim_in, self.fcs = filter_bank.get_fully_connected_layers()
        self._param_list.extend(self.fcs[0].parameters())
        self._param_list.extend(self.fcs[1].parameters())
        
        # remember our dna (so we can mutate it)
        self.dna = dna
    
    def forward(self, x):
        # apply conv/r2r/maxpool blocks
        for layer_id, block_set in self.blocks:
            layer = self.filter_bank.get_layer(layer_id)
            layer_out = self._residual(x, layer['out'], layer['maxpool'])
            for block in block_set:
                layer_out += block(x)    
            x = layer_out
        
        # reshape for fc
        x = x.view(-1, self.fc_dim_in)
        
        # apply fc to get predictions, applying relu between each
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = F.relu(x)
        x = self.fcs[-1](x)
        
        return x
    
    def _get_device(self, tensor):
        if tensor.is_cuda:
            return t.device('cuda')
        else:
            return t.device('cpu')
    
    def _residual(self, x, pad_channels, maxpool):
        """
        Computes the (padded) residual connection, IF there is one
        If not, returns zero
        """
        if maxpool:
            return 0
        
        size = list(x.size())
        size[1] = pad_channels
        buf = t.zeros(size, device=self._get_device(x))
        buf[:,:x.size()[1]] = x
        return buf
    
    def _count_params(self):
        """
        Comput the number of parameters
        
        :return: the number of parameters
        """
        total_num_params = 0
        for parameter in self.parameters():
            num_params = np.prod(t.tensor(parameter.size()).numpy())
            total_num_params += num_params
        return total_num_params
    
    def mutate_dna(self, deeper_prob=0.5, kernel_size_probs={1:0.3, 3:0.35, 5:0.35}):
        """
        Revamp this to actually be good and not massively hacky
        Let it be a little more general with the kernel_size_probs_dict
        """
        uniform_rv = random.random()
        deepen_mutation = True
        if uniform_rv > deeper_prob:
            deepen_mutation = False
            
        uniform_rv = random.random()
        kernel_size = 1 
        if uniform_rv > kernel_size_probs[1]:
            kernel_size = 3
            if uniform_rv > kernel_size_probs[1] + kernel_size_probs[3]:
                kernel_size = 5
                
        # uniformly choose the layer
        layer = random.randint(0, len(self.dna)-1)
        layer_dict = self.filter_bank.get_layer(self.dna[layer][0])
        while layer_dict['maxpool']:
            layer = random.randint(0, len(self.dna)-1)
            layer_dict = self.filter_bank.get_layer(self.dna[layer][0])
            
                
        # make the internal, intermediate channels half that of the output
        out_channels = layer_dict['out']
        intermediate_channels = out_channels #// 2
        
        if deepen_mutation:
            return self._deeper_dna(layer, kernel_size, intermediate_channels, out_channels)
        else:
            return self._wider_dna(layer, kernel_size, intermediate_channels)
    
    def _wider_dna(self, layer, kernel_size, intermediate_channels):
        layer_id = self.dna[layer][0]
        block_id = self.filter_bank.widen_layer(layer_id, kernel_size, intermediate_channels)
        new_dna = copy.deepcopy(self.dna)
        new_dna[layer][1].append(block_id)
        return new_dna
    
    def _deeper_dna(self, layer, kernel_size, intermediate_channels, out_channels):
        # layer index in OUR list to add at
        # insert AFTER the layer specified
        layer_id = self.filter_bank.gen_layer(out_channels, out_channels) # extending from out_channels, also need out_channels
        block_id = self.filter_bank.widen_layer(layer_id, kernel_size, intermediate_channels)
        layer_pair = (layer_id, [block_id])
        new_dna = copy.deepcopy(self.dna)
        new_dna.insert(layer+1, layer_pair)
        return new_dna
        
    
    def print(self):
        """
        Print dna, and print this, so that we can see what the network contains
        """
        pp = pprint.PrettyPrinter(indent=4)
#         pp.pprint("---\nR2R_Cifar_Resnet_object:\n---")
#         pp.pprint(self)
        pp.pprint("\n---\nDNA:\n---")
        pp.pprint(self.dna)
        
        
        
        
        
        
        
        
def _accuracy(prediction, target):
    _, pred_classes = t.max(prediction, 1)
    _, actual_classes = t.max(target, 1)
    return t.mean((pred_classes == actual_classes).type(t.float))
        
    
    
    
    
    
    
    
    
    
    
def _test_loop(model1, dataset, train_iters=300, model2=None):
    epoch_len = 10
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer1 = t.optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)
    
    optimizer2 = None
    if model2 is not None:
        optimizer2 = t.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)
        
    
    for i in range(train_iters+1):
        xs, ys = dataset.next_batch(32)
        ys_pred = model1(xs)
        loss = loss_fn(ys_pred, ys)
        acc1 = _accuracy(ys_pred, ys)
        
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        
        acc2 = 0.0
        if optimizer2 is not None:
            ys_pred = model2(xs)
            loss = loss_fn(ys_pred, ys)
            acc2 = _accuracy(ys_pred, ys)
            
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            
        if i % epoch_len == 0:
            # training accuracy
            print("Iter %d: model one loss - %0.6f, model two loss - %0.6f" % (i, acc1, acc2))
                  
                  
                  
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
def _train_step(model, optimizer, loss_fn, dataset):
    xs, ys = dataset.next_batch()
    ys_pred = model(xs)
    loss = loss_fn(ys_pred, ys)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
def _print_val_perf(model, dataset, i, message):
    xs, ys = dataset.next_val_batch()
    ys_pred = model(xs)
    acc = _accuracy(ys_pred, ys)
    print(message % (i, acc))
    
    
def _val_set_acc(model, dataset):
    accs = []
    for xs, ys in dataset.val_set(50):
        accs.append(_accuracy(model(xs), ys))
    return np.mean(accs)


def _test_set_acc(model, dataset):
    accs = []
    for xs, ys in dataset.test_set(50):
        accs.append(_accuracy(model(xs), ys))
    return np.mean(accs)


def _select_best(model_set, optimizers, dataset, num_to_select):
    scores = [(_val_set_acc(model,dataset), model) for model in model_set]
    scores.sort(key=lambda score_model_pair: score_model_pair[0])
    scores.reverse()
    new_model_set = set([model for _, model in scores[:num_to_select]])
    new_optimizer_map = {model: optimizers[model] for model in new_model_set}
    return new_model_set, new_optimizer_map

def _sub_profile(prediction, target):
    _, pred_classes = t.max(prediction, 1)
    _, actual_classes = t.max(target, 1)
    return (pred_classes == actual_classes).type(t.float).cpu().numpy()

def _validation_profile(model, dataset):
    profile = []
    for xs, ys in dataset.val_set(50):
        profile.extend(_sub_profile(model(xs), ys))
    return profile

def _test_profile(model, dataset):
    profile = []
    for xs, ys in dataset.test_set(50):
        profile.extend(_sub_profile(model(xs), ys))
    return profile

    
    
def _mutate_random(model_set, optimizers, filter_bank, population_size):
    while len(model_set) < population_size:
        rand_model = random.sample(model_set, 1)[0]
        old_num_params = rand_model._count_params()
        new_model = R2R_Cifar_Resnet(filter_bank, rand_model.mutate_dna())
        if next(rand_model.parameters()).is_cuda:
            new_model = new_model.cuda()
        new_num_params = new_model._count_params()
        
        ratio = old_num_params/new_num_params
        lr = optimizers[rand_model].defaults['lr'] * ratio
        weight_decay = optimizers[rand_model].defaults['weight_decay'] * ratio 
        
        model_set.add(new_model)
        optimizers[new_model] = t.optim.Adam(new_model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        
    return model_set, optimizers

    
def _log_accuracies(model, dataset, i, train_accs, val_accs, acc_times, epoch_len):
    xs, ys = dataset.next_batch()
    ys_pred = model(xs)
    train_acc = _accuracy(ys_pred, ys)
    
    xs, ys = dataset.next_val_batch()
    ys_pred = model(xs)
    val_acc = _accuracy(ys_pred, ys)
    
    train_accs.append(train_acc)
    val_accs.append(val_acc)
        
    if len(acc_times) == 0:
        acc_times.append(0)
    else:
        next_time = acc_times[-1] + epoch_len
        acc_times.append(next_time)

        
        
        
        
    
def _neuroevolution_trainer(select, mutate, initial_updates=8000, updates_per_evolution=3000, 
                            population_size=10, selection_size=3, evolutions=15, lr=3e-3, weight_decay=1e-6,
                            eval_freq=50, eval_init_on_test_every_thousand=False):
    """
    TODO: put all of the information we really want about each model in a struct
    E.g. we have a model struct, with struct.model, struct.optimizer, struct.lr, struct.weight_decay
    Rather than keeping these silly maps...
    """
    train_acc = []
    val_acc = []
    acc_times = []
    test_accs = []
    
    epoch_len = eval_freq
    dataset = DatasetCudaWrapper(CifarDataset(64))
    loss_fn = nn.BCEWithLogitsLoss()
    
    fb = R2R_filter_bank()
    model = R2R_Cifar_Resnet(fb, fb.get_initial_dna())
    model = model.cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    
    for i in range(initial_updates+1): 
        _train_step(model, optimizer, loss_fn, dataset)
        if i % epoch_len == 0:
            _print_val_perf(model, dataset, i, "Initial train, iter %d, init model's accuracy: %0.6f")
            _log_accuracies(model, dataset, i, train_acc, val_acc, acc_times, epoch_len)
        if i % 1000 == 0:
            test_acc = _test_set_acc(model, dataset)
            test_accs.append(test_acc)
            
            
    model_set = set([model])
    optimizers = {model: optimizer}
    
    for evolution in range(evolutions):
        model_set, optimizers = select(model_set, optimizers, dataset, selection_size)
        model_set, optimizers = mutate(model_set, optimizers, fb, population_size)
        for i in range(updates_per_evolution):
            model = random.sample(model_set, 1)[0]
            _train_step(model, optimizers[model], loss_fn, dataset)
            if i % epoch_len == 0:
                _print_val_perf(model, dataset, i, "Evo %d. " % evolution + "Iter %d. Random model's accuracy: %0.6f")
                _log_accuracies(model, dataset, i, train_acc, val_acc, acc_times, epoch_len)
    
    # compute evo times
    evo_times = [initial_updates + k*updates_per_evolution for k in range(evolutions)]
    
    # only want to return the best models
    model_set, _ = select(model_set, optimizers, dataset, 4)
        
    return model_set, train_acc, val_acc, acc_times, evo_times, test_accs, fb










def _select_diverse(model_set, optimizers, dataset, num_to_select):
    profiles = {}
    for model in model_set:
        profiles[model] = _validation_profile(model, dataset)
        
    best_models = []
    
    # pick initial, best model
    scores = {}
    best_score = -1.0
    best_model = None
    for model in model_set:
        score = np.mean(profiles[model])
        if score > best_score:
            best_score = score
            best_model = model
    
    best_models.append(best_model)
    model_set.remove(best_model)
    
    # for the remaining, compute weights over the profiles
    length = len(profiles[best_model])
    
    for _ in range(1,num_to_select):
        if len(model_set) == 0:
            break
        
        weights = np.zeros(length)
        for model in best_models:
            weights += np.array(profiles[model])
        
        weights = np.power(weights + 1, -2)
        
        scores = {}
        best_score = -1.0
        best_model = None
        for model in model_set:
            score = np.mean(np.array(profiles[model]) * weights)
            if score > best_score:
                best_score = score
                best_model = model
        
        best_models.append(best_model)
        model_set.remove(best_model)
    
    new_optimizer_map = {model: optimizers[model] for model in best_models}
    return set(best_models), new_optimizer_map







        
        
         
        
        

class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 40)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        
    def build_input(self, model_list, x):
        assert(len(model_list)==4)
        model_outs = []
        for model in model_list:
            model_outs.append(model(x))
        return t.cat(model_outs)
    
    
    
def ensemble_train_loop(model_list, dataset, train_iters=6000):
    ensemble = Ensemble()
    ensemble = ensemble.cuda()
    
    epoch_len = 50
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = t.optim.Adam(ensemble.parameters(), lr=1e-5, weight_decay=0.0, amsgrad=True)
    
    def ensmbl(x):
        return ensemble(ensemble.build_input(model_list, x))
    
    train_accs = []
    val_accs = []
    time = []
    
    for i in range(train_iters+1):
        xs, ys = dataset.next_batch(32)
        ys_pred = ensmbl(xs)

        loss = loss_fn(ys_pred, ys)
        acc = _accuracy(ys_pred, ys)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % epoch_len == 0:
            time.append(i)
            train_accs.append(acc)
            
            xs, ys = dataset.next_val_batch(32)
            ys_pred = ensmbl(xs)
            loss = loss_fn(ys_pred, ys)
            acc = _accuracy(ys_pred, ys)
            
            val_accs.append(acc)
            
            _print_val_perf(ensmbl, dataset, i, "Ensemble training iter %d, val acc %0.6f.")
    
    test_acc = _test_set_acc(ensmbl, dataset)
    
    return train_accs, val_accs, time, test_acc
















def baseline_test():
    results = _neuroevolution_trainer(_select_best, 
                                      _mutate_random, 
                                      initial_updates=10000, 
                                      evolutions=0, 
                                      lr=3e-3, 
                                      weight_decay=1e-6,
                                      eval_freq=50, 
                                      eval_init_on_test_every_thousand=True)
    _, train_acc, val_acc, acc_times, _, test_accs, _ = results
    
    save_dict = {"train_acc": train_acc,
                 "val_acc": val_acc,
                 "acc_times": acc_times,
                 "test_accs": test_accs}
    save(save_dict, "baseline_ne_tests")
    
    

def ne_test():
    results = _neuroevolution_trainer(_select_best, 
                                      _mutate_random, 
                                      initial_updates=1000, 
                                      updates_per_evolution=5000,
                                      evolutions=8, 
                                      lr=3e-3, 
                                      weight_decay=1e-5,
                                      eval_freq=50)
    model_set, train_acc, val_acc, acc_times, evo_times, _, filter_bank = results
    
    dataset = DatasetCudaWrapper(CifarDataset(64))
    val_profiles = []
    test_profiles = []
    test_accs = []
    for model in model_set:
        val_profiles.append(_validation_profile(model, dataset))
        test_profiles.append(_test_profile(model, dataset))
        test_accs.append(_test_set_acc(model, dataset))
        
    
    print("\n\n\n\n")
    print("---\nFilterBank\n---\n")
    filter_bank.print()
    print("\n\n\n\n")
    print("---\nPrinting Best Models\n---\n")
    for model in model_set:
        model.print()
    
    
    save_dict = {"train_acc": train_acc,
                 "val_acc": val_acc,
                 "acc_times": acc_times,
                 "evo_times": evo_times,
                 "val_profiles": val_profiles,
                 "test_profiles": test_profiles,
                 "test_accs": test_accs}
    save(save_dict, "ne_tests")
    
    
    
    
    
    
    
    
    
    
    
    
def ensemble_test():
    results = _neuroevolution_trainer(_select_diverse, 
                                      _mutate_random, 
                                      initial_updates=1000, 
                                      updates_per_evolution=5000,
                                      evolutions=10, 
                                      lr=1e-3, 
                                      weight_decay=1e-4,
                                      eval_freq=50)
    model_set, train_acc, val_acc, acc_times, evo_times, _, filter_bank = results
    
    dataset = DatasetCudaWrapper(CifarDataset(64))
    val_profiles = []
    test_profiles = []
    test_accs = []
    for model in model_set:
        val_profiles.append(_validation_profile(model, dataset))
        test_profiles.append(_test_profile(model, dataset))
        test_accs.append(_test_set_acc(model, dataset))
        
        
    print("\n\n\n\n")
    print("---\nFilterBank\n---\n")
    filter_bank.print()
    print("\n\n\n\n")
    print("---\nPrinting Best Models\n---\n")
    for model in model_set:
        model.print()
        
    #
    print('\n'*5)
    print("IT'S ABOUT ENSEMBLE TIME, I'M SICK OF LEARNING LIKE CRAP")
    
    model_list = list(model_set)
    e_train_accs, e_val_accs, e_time, e_test_acc = ensemble_train_loop(model_list, dataset, train_iters=5000)
    
    save_dict = {"train_acc": train_acc,
                 "val_acc": val_acc,
                 "acc_times": acc_times,
                 "evo_times": evo_times,
                 "val_profiles": val_profiles,
                 "test_profiles": test_profiles,
                 "test_accs": test_accs,
                 "ens_train_accs": e_train_accs,
                 "ens_val_accs": e_val_accs,
                 "ens_time": e_time,
                 "ens_test_acc": e_test_acc}
    save(save_dict, "coevolve_ne_tests")
    
        
        
        
        
        
        
        

if __name__ == "__main__":
    # sanity checks
    r2r_conv_sanity = False
    r2r_block_sanity = False
    filter_bank_sanity = False
    cifar_resnet_sanity = False
    cifar_resnet_mutate_sanity = False
    printing_sanity = False
    coevolution_sanity = True
    
                
    if r2r_conv_sanity:
        conv = R2R_conv(1, 2, kernel_size=3, padding=1,
                        adapt_for_repeated_in_channels=False, repeat_new_output_channels=True)
        x = t.tensor([[[[1.0, 2.0], [-2.0, 1.0]]]])
        print(conv(x))
        x = t.tensor([[[[1.0, 2.0], 
                        [-2.0, 1.0]],
                       [[1.0, 2.0],
                        [-2.0, 1.0]]]])
        conv = R2R_conv(2, 2, kernel_size=3, padding=1,
                        adapt_for_repeated_in_channels=True, repeat_new_output_channels=False)
        print(conv(x))
                
    if r2r_block_sanity:
        block = R2R_block(2, 2, 4, 3)
        x = t.tensor([[[[1.0, 2.0], 
                        [-2.0, 1.0]],
                       [[1.0, 2.0],
                        [-2.0, 1.0]]]])
        print(block(x))
    
    if filter_bank_sanity:
        fb = R2R_filter_bank(cuda=False)
        print(fb.get_initial_dna())
        fb.widen_layer(2, 5, 60)
        print(fb.get_layer(2))
        #fb.widen_layer(1, 5, 60) # should be an error, if uncommented
        print(fb.layers)
        
    if cifar_resnet_sanity:
        dataset = CifarDataset(32)
        fb = R2R_filter_bank(cuda=False)
        model = R2R_Cifar_Resnet(fb, fb.get_initial_dna())
        _test_loop(model, dataset) 
        
        # widen, and manually make a crazier network, train in parallel, and check that they don't die
        fb.widen_layer(0, 5, 8)
        fb.widen_layer(2, 5, 16)
        fb.widen_layer(4, 5, 32)
        print(fb.layers)
        
        all_blocks_dna = [(0, [0,1]),
                          (1, [0]),
                          (2, [0,1]),
                          (3, [0]),
                          (4, [0,1]),
                          (5, [0])]
        model2 = R2R_Cifar_Resnet(fb, all_blocks_dna)
        _test_loop(model, dataset, model2=model2)
        
    if cifar_resnet_mutate_sanity:
        dataset = CifarDataset(32)
        fb = R2R_filter_bank(cuda=False)
        model = R2R_Cifar_Resnet(fb, fb.get_initial_dna())
        print(model._count_params())
        
        print(model.dna)
        deeper_dna = model._deeper_dna(0,5,10, 16)
        print(deeper_dna)
        wider_dna = model._wider_dna(2,1,10)
        print(wider_dna)
        
        wider_model = R2R_Cifar_Resnet(fb, wider_dna)
        print(wider_model._count_params())
        
        for _ in range(20):
            print(model.mutate_dna())
            
        xs, ys = dataset.next_batch(2)
        print(wider_model(xs)) # just a sanity that a mutated model works!
        
    if printing_sanity:
        fb = R2R_filter_bank(cuda=False)
        model = R2R_Cifar_Resnet(fb, fb.get_initial_dna())
        deeper_dna = model._deeper_dna(0,5,10, 16)
        wider_dna = model._wider_dna(2,1,10)
        
        print(fb.print())
        print(model.print())
        
    if coevolution_sanity:
        pass # TODO
    
    
    
    
    run_baseline_test = False
    run_ne_test = False
    run_ensemble_test = True
    
    
    
    
    if run_baseline_test:
        baseline_test()
    
    if run_ne_test:
        ne_test()
        
    if run_ensemble_test:
        ensemble_test()
        
        
#     best_models = _neuroevolution_trainer(_select_best, _mutate_random, 
#                             initial_updates=50, updates_per_evolution=50, 
#                             population_size=5, selection_size=2, evolutions=3, lr=3e-3, weight_decay=1e-6)
#     print(best_models)
#     for model in best_models:
#         model.print()