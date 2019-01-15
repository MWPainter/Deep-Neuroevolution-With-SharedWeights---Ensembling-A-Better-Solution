import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from r2r.init_utils import *
from r2r.module_utils import *
from utils.pytorch_utils import cudafy
from utils.plotting_utils import count_parameters_in_list

import copy
import math
from itertools import chain





"""
This file contains all of the logic related to performing widening and deepening transforms. 

We define a number of interfaces that need to be implemented by a nn.Module in order to use the R2R functions.

We use the following python and PyTorch conventions for function naming very explicitly:
_function_name = a private/helper function, not to be exported;
function_name_ = a function that has side effects (will change the objects that have been passed into them).

Note, that this means that we will sometimes be declaring function names like '_function_name_', when both cases are true.

The three functions that a user should look at potentially using for their function preserving widening/deepening needs 
are:
- r_2_wider_r_              - widen a single hidden volume
- widen_network_            - widen all hidden volumes in a network
- make_deeper_network_      - deepen a network




R2WiderR interface (required by nn.Modules that wish to use R2WiderR):
- network_instance.hvg()
    This returns a HVG object that describes the 'Hidden Volume Graph' defined by the network. See our examples 
    and the definition of the HVG object for how to use this. (Todo at some point: add some sort of diagram).
Example implementations: Mnist_Resnet, Cifar_Resnet
    
    

    
R2DeeperR interface (required by nn.Modules that wish to use R2DeeperR):
- network_instance.conv_forward(x) 
    The convolutional portion of forward for the nn.Module
- network_instance.fc_forward(x)
    The fully connected portion of forward for the nn.Module
- network_instance.out_forward(x)
    The output portion of forward for the nn.Module
- network_instance.forward()
    Should be equal to network_instance.out_forward(network_instance.fc_forward(network_instance.conv_forward(x)))
Example implementations: Mnist_Resnet, Cifar_Resnet




R2R interface (required by nn.Modules that wish to use R2DeeperR and R2WiderR):
- all of R2WiderR interface, where:
    - network_instance.hvg()
        Should just chain together the building of conv_hvg() and fc_hvg() below. (See examples).
- all of R2DeeperR interface
- network_instance.input_shape()
    Returns the input shape to the network.
- network_instance.conv_hvg(cur_hvg)
    If the network instance implements the hvg function for R2WiderR, then is should implement building the conv part 
    of the hvg with this method, given the 'cur_hvg', which should just consist of the input volume. This function 
    should return the resulting hvg.
- network_instance.fc_hvg(cur_hvg)
    If the network instance implements the hvg function for R2WiderR, then is should implement building the fully 
    connected part of the hvg with this method, given the 'cur_hvg', which is the hvg output by the conv_hvg part. 
    This function should return the resulting hvg.
Example implementations: Mnist_Resnet, Cifar_Resnet
    
    
    
    
R2DeeperRBlock interface (required by any nn.Module that wishes to be the block used for extending in R2DeeperR):
- identity initialization
    The nn.Module should be initialized such that it is an identity operation if the transform is wished to be 
    function preserving. (Recommended that the initializer takes an argument that can set if it's identity initialized 
    or not).
- block.conv_hvg(cur_hvg)
    The nn.Module should implement a conv_hvg function, for the hvg interface, so that it can be used in future widening 
    operations, where the 'base_network' implemented the hvg interface. This should 
- 
Example implementations: Res_Block
"""





# Only export things that actually widen/deepen volumes, and not helper functions
__all__ = ['r_2_wider_r_',
           'HVG', 
           'HVN', 
           'HVE', 
           'widen_network_',
           'Deepened_Network',
           'make_deeper_network_'] # make_deeper_network = r2deeperr if add identity initialized module





"""
Some helper functions :)
"""




def _is_linear_volume_shape(vol_shape):
    """
    Checks if a hidden volume is a "linear" volume
    """
    return len(vol_shape) == 1





def _is_conv_volume_shape(vol_shape):
    """
    Checks if a hidden volume is a "convolutional" volume
    """
    return len(vol_shape) == 3





def _round_up_multiply(a, b, m):
    """
    Performs a*b and rounds up to the nearest m. Note that (x+m-1)//m, is a divide by m rounded up
    """
    prod = int(math.ceil(((a*b + m - 1) // m) * m))
    return prod





"""
Widening hidden volumes
"""




def r_2_wider_r_(prev_layers, volume_shape, next_layers, batch_norm=None, residual_connection=None, extra_channels=0,
                 init_type="match_std", function_preserving=True, multiplicative_widen=True, mfactor=2, net_morph=False,
                 net_morph_add_noise=True):
    """
    Single interface for r2widerr transforms, where prev_layers and next_layers could be a single layer.

    'volume_shape' refers to the shape that we wish to widen, and 'prev_layers' and 'next_layers' are the nn.Modules
    that we will edit to widen in a function preserving way.

    At a high level, we are considering the local operation of next(phi(prev(...))), where phi is any channel wise
    function. phi can incorporate batch norm, which is fed into the 'batch_norm' input. In the paper, phi may involve
    an operation that reduces the spatial dimension of the hidden volume, such as max pooling, however, in the
    implementation, we consider pooling layers explicitly as 'next', and propagate channel and concatination information
    through the pooling layers using a cache. This is necessary to deal with the pooling layers used as 'subnetweorks'
    in inception style networks, which aren't explicitly considered in the paper.
    
    For now, with convolutions, we only support variable sizes kernels, strides and padding. We don't support 
    any non-default padding and so on.
    
    Finally, we allow ourselves to initialize the new varialbles in such a way that the functions output is 
    preserved. This is implemented using the 'alpha' variable of the underlying functions that initialize the 
    new weights.
    
    For now we support the following *ordered* pairs of nn.Modules:
    (nn.Conv2d, nn.Conv2d)
    (nn.Conv2d, nn.Linear)
    (nn.Linear, nn.Linear)

    Additionally we support modules of type nn.MaxPool2d and nn.AvgPool2d, however, for the overall transformation
    to be made function preserving (or just flatout work with respect to shaping), any time a pooling layer is seen
    in next_layers, it must then be seen in prev_layers in another . Specifically, any volume that is computed
    using the output of a "widened maxpool/avgpool" must also be widened to actually perform the negations needed to
    make the transform function preserving.

    Note that the consideration of avg/max pools in the paper is as part of the activation function, and that is why
    it is necessary to cache information in the pooling module object, to "skip" a layer.

    :param prev_layers: A (list of) layer(s) before the hidden volume/units being widened (their outputs are 
            concatenated together)
    :param volume_shape: The shape of the volume for which we wish to widen
    :param next_layers: A (list of) layer(s) layer after the hidden volume/units being widened (inputs are the 
            concatenated out)
    :param batch_norm: The batch norm function associated with the volume being widened (None if there is no batch norm)
    :param residual_connection: The Residual_Connection object associated with this volume. (I.e. if this volume x is
            used for a residual connection at some point later in the network y, with y = residual_connection(y,x)).
            This should be none if it's not used as part of a residual connection.
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param function_preserving: If we wish for the widening to preserve the function I/O
    :param multiplicative_widen: If we want the number of extra channels to be multiplicative (i.e. 2 = double number
            of layers) or additive (i.e. 2 = add two layers)
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    :param net_morph: If we wish to provide a netmorph widening (rather than R2R widening).
    :param net_morph_add_noise: If we are using net morph, then add noise to the filters that are set to zero.
    """
    # Handle inputting of single input/output layers
    if type(prev_layers) != list:
        prev_layers = [prev_layers]
    if type(next_layers) != list:
        next_layers = [next_layers]
        
    _r_2_wider_r_(prev_layers, volume_shape, next_layers, batch_norm, residual_connection, extra_channels, init_type,
                  function_preserving, multiplicative_widen, mfactor, net_morph, net_morph_add_noise)




def _r_2_wider_r_(prev_layers, volume_shape, next_layers, batch_norm, residual_connection, extra_channels, init_type,
                  function_preserving, multiplicative_widen, mfactor, net_morph, net_morph_add_noise):
    """  
    The full internal implementation of r_2_wider_r_. See description of r_2_wider_r_.
    More helper functions are used.
    """
    # For us to be able to perform a function preserving transforms, extra channels must be even and non-negative
    if extra_channels <= 0 or (function_preserving and not multiplicative_widen and extra_channels % 2 != 0):
        raise Exception("Invalid number of extra channels in widen.")
    
    # Check that the types of all the in/out layers are the same
    prev_type = nn.Linear if any([type(layer) == nn.Linear for layer in prev_layers]) else nn.Conv2d
    next_type = nn.Linear if any([type(layer) == nn.Linear for layer in next_layers]) else nn.Conv2d
    for i in range(1,len(prev_layers)):
        if type(prev_layers[i]) not in [prev_type, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
            raise Exception("All (non pooling) input layers in R2WiderR need to be the same type, nn.Conv2D or nn.Linear")
    for i in range(1,len(next_layers)):
        if type(next_layers[i]) not in [next_type, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
            raise Exception("All (non pooling) output layers in R2WiderR need to be the same type, nn.Conv2D or nn.Linear")
    for i in range(1,len(prev_layers)):
        if type(prev_layers[i]) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d] and not hasattr(prev_layers[i], 'r2r_cache'):
            raise Exception("AvgPool2d or MaxPool2d in prev_layers when not used previously in a R2WiderR call in "
                            "next_layers, which is necessary to create a cache of values needed for the transformation.")

    # Check that the volume shape is either linear or convolutional
    if len(volume_shape) not in [1,3]:
        raise Exception("Volume shape must be 1D or 3D for R2WiderR to work")
            
    # Get if we have a linear input or not
    input_is_linear = any([type(prev_layer) == nn.Linear for prev_layer in prev_layers])

    # Work out the number of hiden units per new channel 
    # (for fc pretend 1x1 spatial resolution, so 1 per channel) (for conv this is width*height)
    # For conv -> linear layers this can be a little complex. But we always know the number of channels from the prev kernels
    channels_in_volume = np.sum([_get_output_channels_from_layer(layer) for layer in prev_layers])
    total_hidden_units = np.prod(volume_shape)
    new_hidden_units_in_next_layer_per_new_channel = total_hidden_units // channels_in_volume
    
    # Sanity checks
    if input_is_linear and new_hidden_units_in_next_layer_per_new_channel != 1:
        raise Exception("Number of 'new hidden_units per new channel' must be 1 for linear. Something went wrong :(.")
    
    # Compute the slicing of the volume from the input (to widen outputs appropraitely)
    volume_slices_indices = [0]
    module_slices_indices = [0]
    for prev_layer in prev_layers:
        base_index = volume_slices_indices[-1]
        new_slice_indices = _compute_new_volume_slices_from_layer(base_index, prev_layer)
        volume_slices_indices.extend(new_slice_indices)
        module_slices_indices.append(new_slice_indices[-1])
    
    # Sanity check that our slices are covering the entire volume that is being widened
    # There is some complexity when conv volumes are flattened as input to a linear layer
    # We effectively check that the input is consistent with the volume shape here
    if ((_is_conv_volume_shape(volume_shape) and volume_slices_indices[-1] != volume_shape[0]) or
        (_is_linear_volume_shape(volume_shape) and volume_slices_indices[-1] * new_hidden_units_in_next_layer_per_new_channel != volume_shape[0])):
        raise Exception("The shape output from the input layers is inconsistent with the hidden volume provided in R2WiderR.")

    # Net morph sets the sparser
    # This is unused by R2R portion of this subroutine
    is_prev_layer_sparse = False
    is_next_layer_sparse = False
    if net_morph and function_preserving:
        prev_layer_params = count_parameters_in_list(prev_layers)
        next_layer_params = count_parameters_in_list(next_layers)
        if _contains_pooling_module(next_layers): # If next layers contains a pooling layer,
            is_prev_layer_sparse = True
        elif prev_layer_params < next_layer_params or residual_connection is not None:
            is_prev_layer_sparse = True
        else:
            is_next_layer_sparse = True
    
    # Iterate through all of the prev layers, and widen them appropraitely
    for prev_layer in prev_layers:
        _widen_output_channels_(prev_layer, extra_channels, init_type, multiplicative_widen, input_is_linear,
                                    function_preserving, mfactor, net_morph, is_prev_layer_sparse)
    
    # Widen batch norm appropriately 
    if batch_norm:
        _extend_bn_(batch_norm, extra_channels, module_slices_indices, multiplicative_widen, mfactor, net_morph)

    # Widen the residual connection appropriately (and don't for function preserving net morph)
    if residual_connection:
        alpha = 1.0 if not function_preserving else -1.0
        residual_connection._widen_(volume_slices_indices, extra_channels, multiplicative_widen, alpha, mfactor)
    
    # Iterate through all of the next layers, and widen them appropriately. (Needs the slicing information to deal with concat)
    for next_layer in next_layers:
        _widen_input_channels_(next_layer, extra_channels, init_type, volume_slices_indices, input_is_linear,
                                   new_hidden_units_in_next_layer_per_new_channel, multiplicative_widen,
                                   function_preserving, mfactor, net_morph, is_next_layer_sparse)




def _contains_pooling_module(modules):
    """Given a list of nn.Module's, returns true if any are a pooling layer"""
    for module in modules:
        if type(module) in [nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d]:
            return True
    return False





def _get_output_channels_from_layer(layer):
    """
    Gets output channels from a layer, which can be of type nn.Linear, nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d.
    To get the output layers from one of the pooling layers, it has to have been used as part of "next_layers" in a
    previous R2WiderR call.
    """
    if type(layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d] and hasattr(layer, 'r2r_old_channels_propagated'):
        return layer.r2r_old_channels_propagated
    elif type(layer) in [nn.Conv2d, nn.Linear]:
        return layer.weight.size(0)
    else:
        raise Exception("Couldn't get output channels from layer.")





def _compute_new_volume_slices_from_layer(base_index, prev_layer):
    """
    Computes the new slicing indices (in the current volume of consideration) contributed from the layer 'prev_layer'.
    For example, if prev_layer is a convolutional layer, with 10 ouput channels, and the base index is 20, then we
    should output [30]. This is because this convolutional layer is contributing to layer 20-29 in the volume, and,
    the base index (20) has already been considered.

    The situation is more complex when considering pooling layer, which, may take an input that was a concatination from
    many layers. This slciing information needs to be propagated through the pooling layer from the input.

    :param base_index: The current index that we have considered up to
    :param prev_layer: A layer from prev_layers in R2WiderR who's output makes part of the volume being widened.
    :return: New indices into the volume being widened, contributed from 'prev_layer'
    """
    if type(prev_layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d] and hasattr(prev_layer, 'r2r_volume_slice_indices'):
        propagating_slice_indices = prev_layer.r2r_volume_slice_indices[1:]
        return [base_index + index for index in propagating_slice_indices]
    elif type(prev_layer) in [nn.Conv2d, nn.Linear]:
        return [base_index + prev_layer.weight.size(0)]
    else:
        raise Exception("Couldn't get new volume slice indices from some layer in prev_layers.")
        





def _extend_bn_(bn, new_channels_per_slice, module_slices_indices, multiplicative_widen, mfactor, net_morph):
    """
    Extend batch norm with 'new_channels' many extra units. Initialize values to zeros and ones appropriately.
    Really this is just a helper function for R2WiderR
    
    :param bn: The batch norm layer to be extended (or list of batch norms)
    :param new_channels_per_slice: The number of new channels to add, per slice
    :param module_slices_indices: The indices to be able to slice the volume per module input (so that we can extend
            batch norm(s) appropriately for a concatinated input).
    :param multiplicative_widen: If true, then we should interpret "extra channels" as a multiplicative factor for the
            number of outputs (rather than additive)
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    :param net_morph: If we actually would like to use the NetMorph widening, rather than the R2R widening.
    """
    # Sanity checks
    bn_is_array = hasattr(bn, '__len__')
    if hasattr(bn, '__len__') and len(bn) + 1 != len(module_slices_indices):
        raise Exception("Number of batch norms and volume slice indices inconsistent.")

    # Get the old scale/shift/mean/var as a single vector
    new_scale_slices = []
    new_shift_slices = []
    new_running_mean_slices = []
    new_running_var_slices = []

    if not bn_is_array:
        old_scale = bn.weight.data.cpu().numpy()
        old_shift = bn.bias.data.cpu().numpy()
        old_running_mean = bn.running_mean.data.cpu().numpy()
        old_running_var = bn.running_var.data.cpu().numpy()
    else:
        total_channels = module_slices_indices[-1]
        old_scale = np.ones(total_channels)
        old_shift = np.zeros(total_channels)
        old_running_mean = np.zeros(total_channels)
        old_running_var = np.ones(total_channels)

        for i in range(0,len(module_slices_indices)-1):
            if bn[i] is None:
                continue
            beg = module_slices_indices[i]
            end = module_slices_indices[i+1]
            old_scale[beg:end] = bn[i].weight.data.cpu().numpy()
            old_shift[beg:end] = bn[i].bias.data.cpu().numpy()
            old_running_mean[beg:end] = bn[i].running_mean.data.cpu().numpy()
            old_running_var[beg:end] = bn[i].running_var.data.cpu().numpy()

    # Extend the bn vector
    for i in range(0,len(module_slices_indices)-1):
        beg = module_slices_indices[i]
        end = module_slices_indices[i+1]
        
        # to triple number of channels, *add* 2x the current num
        new_channels = new_channels_per_slice if not multiplicative_widen else _round_up_multiply(new_channels_per_slice - 1, (end-beg), mfactor)

        if net_morph:
            new_scale_slices.append(_zero_pad_1d(old_scale[beg:end], new_channels))
            new_shift_slices.append(_zero_pad_1d(old_shift[beg:end], new_channels))
        else:
            new_scale_slices.append(_mean_pad_1d(old_scale[beg:end], new_channels))
            new_shift_slices.append(_mean_pad_1d(old_shift[beg:end], new_channels))
        new_running_mean_slices.append(_zero_pad_1d(old_running_mean[beg:end], new_channels))
        new_running_var_slices.append(_one_pad_1d(old_running_var[beg:end], new_channels))

    # Assign to batch norm(s)
    if not bn_is_array:
        new_scale = np.concatenate(new_scale_slices)
        new_shift = np.concatenate(new_shift_slices)
        new_running_mean = np.concatenate(new_running_mean_slices)
        new_running_var = np.concatenate(new_running_var_slices)
        _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)

    else:
        for i in range(len(bn)):
            if bn[i] is None:
                continue
            _assign_to_batch_norm_(bn[i], new_scale_slices[i], new_shift_slices[i], new_running_mean_slices[i],
                                   new_running_var_slices[i])

    
    
    
    
    
def _widen_output_channels_(prev_layer, extra_channels, init_type, multiplicative_widen, input_is_linear,
                            function_preserving, mfactor, net_morph, is_prev_layer_sparse):
    """
    Helper function for r2widerr. Containing all of the logic for widening the output channels of the 'prev_layers'.
    
    :param prev_layer: A layer before the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param multiplicative_widen: If true, then we should interpret "extra channels" as a multiplicative factor for the
            number of outputs, rather than additive
    :param input_is_linear: If input is from a linear layer previously
    :param function_preserving: If we want the widening to be done in a function preserving fashion
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    :param net_morph: If we actually would like to use the NetMorph widening, rather than the R2R widening.
    :param is_prev_layer_sparse: If the prev layers are sparser than next layers (only used for net morph)
    """
    if type(prev_layer) is nn.Conv2d:
        # If we have a bias in the conv
        module_has_bias = prev_layer.bias is not None

        # unpack conv2d params to numpy tensors
        prev_kernel = prev_layer.weight.data.cpu().numpy()
        prev_bias = None
        if module_has_bias:
            prev_bias = prev_layer.bias.data.cpu().numpy()
        
        # new conv kernel
        old_out_channels, in_channels, height, width = prev_kernel.shape
        if multiplicative_widen:
            extra_channels = _round_up_multiply(old_out_channels, (extra_channels - 1), mfactor) # to triple number of channels, *add* 2x the current num
        kernel_extra_shape = (extra_channels, in_channels, height, width)

        if not net_morph:
            prev_kernel = _extend_filter_with_repeated_out_channels(kernel_extra_shape, prev_kernel, init_type)
        else:
            prev_kernel_init_type = init_type if not is_prev_layer_sparse else 'zero'
            prev_kernel = _extend_filter_out_channels(kernel_extra_shape, prev_kernel, prev_kernel_init_type)

        # zero pad the bias
        if module_has_bias:
            prev_bias = _zero_pad_1d(prev_bias, extra_channels)
        
        # assign new conv and bias
        _assign_kernel_and_bias_to_conv_(prev_layer, prev_kernel, prev_bias)
        
    elif type(prev_layer) is nn.Linear:
        # If we have a bias in the linear
        module_has_bias = prev_layer.bias is not None

        # unpack linear params to numpy tensors
        prev_matrix = prev_layer.weight.data.cpu().numpy()
        prev_bias = None
        if module_has_bias:
            prev_bias = prev_layer.bias.data.cpu().numpy()
        
        # work out the shape of what we need to add
        old_n_out, n_in = prev_matrix.shape
        if multiplicative_widen:
            extra_channels = _round_up_multiply(old_n_out, (extra_channels - 1), mfactor) # to triple number of channels, *add* 2x the current num
        matrix_extra_shape = (extra_channels, n_in)

        # Compute the new matrix using the widening method chosen
        if not net_morph:
            prev_matrix = _extend_matrix_with_repeated_out_weights(matrix_extra_shape, prev_matrix, init_type)
        else:
            prev_matrix_init_type = init_type if not is_prev_layer_sparse else 'zero'
            prev_matrix = _extend_matrix_out_weights(matrix_extra_shape, prev_matrix, prev_matrix_init_type)


        # zero pad the bias
        if module_has_bias:
            prev_bias = _zero_pad_1d(prev_bias, extra_channels)
        
        # assign new matrix and bias
        _assign_weights_and_bias_to_linear_(prev_layer, prev_matrix, prev_bias)

    elif type(prev_layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
        # Unpack, and check consistency
        prev_extra_channels, prev_input_is_linear, prev_multiplicative_widen = prev_layer.r2r_cache
        if function_preserving and (prev_extra_channels != extra_channels or prev_multiplicative_widen != multiplicative_widen):
            raise Exception("Widen with a pooling layer in prev_layers inconsistent wiht the call made when it was in "
                            "next_layers, making the cached values invalid for a function presering transform.")
        if prev_input_is_linear != input_is_linear:
            raise Exception("Pooling layer allows accidental mixing of conv and linear layers at the next layer.")

        # Delete the values that we stored in the nn.Module as a cache
        del prev_layer.r2r_cache
        del prev_layer.r2r_old_channels_propagated
        del prev_layer.r2r_volume_slice_indices


    else:
        raise Exception("We can only handle input nn.Modules that are Linear, Conv2d at the moment.")
                            
                            
                            
                            
                            
def _widen_input_channels_(next_layer, extra_channels, init_type, volume_slices_indices, input_is_linear,
                           new_hidden_units_in_next_layer_per_new_channel, multiplicative_widen, function_preserving,
                           mfactor, net_morph, is_next_layer_sparse):
    """
    Helper function for r2widerr. Containing all of the logic for widening the input channels of the 'next_layers'.
    
    :param next_layer: A layer after the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param volume_slices_indices: The indices to be able to slice the volume
    :param input_is_linear: If input is from a linear layer previously
    :param new_hidden_units_in_next_layer_per_new_channel: The number of new hidden units in the volume per channel
            input to 'next_layers'. Conssider a linear 'next_layer' to be a 1x1 conv, on a volume with 1x1 spatial
            dimensions.
    :param multiplicative_widen: If true, then we should interpret "extra channels" as a multiplicative factor for the
            number of outputs, rather than additive
    :param function_preserving: If we want the widening to be done in a function preserving fashion
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    :param net_morph: If we actually would like to use the NetMorph widening, rather than the R2R widening.
    :param is_next_layer_sparse: If the next layers are sparser than prev layers (only used for net morph)
    """
    # Check that we don't do linear -> conv, as haven't worked this out yet
    if input_is_linear and type(next_layer) is nn.Conv2d:
        raise Exception("We currently don't handle the nn.Linear -> nn.Conv2d case in r_2_wider_r.")
    
    if type(next_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        next_kernel = next_layer.weight.data.cpu().numpy()
        
        # Compute the new kernel for 'next_kernel' (extending each slice carefully)
        alpha = -1.0 if function_preserving else 1.0
        out_channels, old_in_channels, height, width = next_kernel.shape
        next_kernel_parts = []
        for i in range(1, len(volume_slices_indices)):
            # Compute the shape of what we need to add
            beg, end = volume_slices_indices[i-1], volume_slices_indices[i]
            slice_extra_channels = extra_channels
            if multiplicative_widen:
                slice_extra_channels = _round_up_multiply((end-beg), (extra_channels - 1), mfactor) # to triple num channels, *add* 2x the current num
            kernel_extra_shape = (out_channels, slice_extra_channels, height, width)

            # Extedn t the kernel according to the widening paradigm we wish to use
            if not net_morph:
                kernel_part = _extend_filter_with_repeated_in_channels(kernel_extra_shape, next_kernel[:,beg:end],
                                                                       init_type, alpha)
            else:
                kernel_part_init_type = init_type if not is_next_layer_sparse else 'zero'
                kernel_part = _extend_filter_in_channels(kernel_extra_shape, next_kernel[:,beg:end], kernel_part_init_type)
            next_kernel_parts.append(kernel_part)

        # Join all of the kernel parts, and then assign new conv (don't need to change bias)
        next_kernel = np.concatenate(next_kernel_parts, axis=1)
        _assign_kernel_and_bias_to_conv_(next_layer, next_kernel)
        
    elif type(next_layer) is nn.Linear:            
        # unpack linear params to numpy tensors
        next_matrix = next_layer.weight.data.cpu().numpy()
        
        # Compute the new matrix for 'next_matrix' (extending each slice carefully)
        alpha = -1.0 if function_preserving else 1.0
        n_out, old_n_in = next_matrix.shape 
        next_matrix_parts = []
        for i in range(1, len(volume_slices_indices)):
            # Compute the shape of what we need to add
            beg = volume_slices_indices[i-1] * new_hidden_units_in_next_layer_per_new_channel
            end = volume_slices_indices[i] * new_hidden_units_in_next_layer_per_new_channel
            extra_params_per_input_layer = extra_channels * new_hidden_units_in_next_layer_per_new_channel
            if multiplicative_widen:
                extra_params_per_input_layer = _round_up_multiply((end-beg), (extra_channels - 1), mfactor) # triple num outputs = *add* 2x the current num
            matrix_extra_shape = (n_out, extra_params_per_input_layer)

            # Extend the matrix according to the widening paradigm we wish to use
            if not net_morph:
                matrix_part = _extend_matrix_with_repeated_in_weights(matrix_extra_shape, next_matrix[:,beg:end],
                                                                      init_type, alpha)
            else:
                matrix_part_init_type = init_type if not is_next_layer_sparse else 'zero'
                matrix_part = _extend_matrix_in_weights(matrix_extra_shape, next_matrix[:,beg:end],
                                                                      matrix_part_init_type)

            # Add this matrix part to the list
            next_matrix_parts.append(matrix_part)

        # Put together all of the parts, and, assign new linear params (don't need to change bias)
        next_matrix = np.concatenate(next_matrix_parts, axis=1)
        _assign_weights_and_bias_to_linear_(next_layer, next_matrix)

    elif type(next_layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
        # If maxpool or avgpool, store the information for when it's used as a 'prev layer'
        # Note that when this max pool is part of "prev_layers" we're pretending that it's 'widened', hence the need to
        # remember the state before
        next_layer.r2r_cache = (extra_channels, input_is_linear, multiplicative_widen)
        next_layer.r2r_old_channels_propagated = volume_slices_indices[-1]
        next_layer.r2r_volume_slice_indices = volume_slices_indices

    else:
        raise Exception("We can only handle output nn.Modules that are Linear, Conv2d, MaxPool2d or AvgPool2d at the moment.")
        


        
        
        






"""
Widening entire networks (by building graphs over the networks)
"""    





class HVG(object):
    """
    Class representing a hidden volume graph.
    
    As nodes have no option to add edges, each node must only be created after all of it's parents. Therefore, we can keep 
    track of the order in which nodes were made to be able to iterate over them correctly (from the nodes representing the 
    networks input, to the nodes representing the networks output).
    """
    def __init__(self, input_shape, residual_connection=None):
        """
        Creates a graph object, with an initial node and no edges.

        :param input_shape: The input shape to the network this will be a HVG for.
        :param residual_connection: The Residual_Connection object associated with the input volume, if any.
                This should be none if the input shape is not used as part of a residual connection.
        """
        self.root_hvn = HVN(input_shape, residual_connection=residual_connection)
        self.nodes = [self.root_hvn]



    def add_hvn_object(self, hvn):
        """
        Adds a HVN ovbject directly to the graph
        :param hvn: A HVN typed object that should be part of this graph
        """
        if hvn in self.nodes:
            raise Exception("Cannot create loops in the HVG.")
        self.nodes.append(hvn)

    
    
    def add_hvn(self, hv_shape, input_modules=[], input_hvns=None, batch_norm=None, residual_connection=None):
        """
        Creates a new hvn node, and is just a wrapper around the HVN constructor. 
        THe wrapper allows the graph object to keep track of the nodes that have been added.
        If input_hvn's aren't specified, assume that its the current set of output nodes from this graph.

        :param hv_shape: The shape of the hidden volume we are adding as a node now
        :param input_modules: A list of nn.Modules that are to be associated to the with the edges to the parent nodes.
                input_modules[i] should correspond to input_hvns[i].
        :param input_hvns: A list of HVN objects in this HVG to be the parent nodes from this volume. If None, then we
                take all of the current 'output nodes' in the graph (those without any children currently).
        :param batch_norm: Any batch norm nn.Module that is associated with this hidden volume.
        :param residual_connection: The Residual_Connection object associated with the volume. (I.e. if this volume x is
                used for a residual connection at some point later in the network y, with y = residual_connection(y,x)).
                This should be none if it's not used as part of a residual connection.
        """
        # If no input hvn's provided, then take all of the current output n odes
        if input_hvns is None:
            input_hvns = self.get_output_nodes()

        # If it's a normal HVN (not pseudo hvn), make a new hvn node and add it to the hvg
        hvn = HVN(hv_shape, input_modules, input_hvns, batch_norm, residual_connection)
        self.nodes.append(hvn)
        return hvn

    

    def node_iterator(self):
        """
        Iterates through the nodes, returning tuples of ([prev_layer_modules], shape, batch_norm,
        residual connection object, [next_layer_modules]), ready to be fed into a volume widening function.

        Note that not all volumes are appropriate to be widened, only the one's that are both output from a layer and input
        to another layer.

        :yields: ([list of parent nn.Modules (edges)], hidden volume shape, is_conv_volume, batch norm,
                    residual connection object, [list of child nn.Modules (edges)]) tuples.
        """
        for node in self.nodes:
            if len(node.parent_edges) != 0 and (len(node.child_edges) != 0 or node.residual_connection is not None):
                yield (node._get_parent_modules(), node.hv_shape, node._is_conv_volume(), node.batch_norm,
                       node.residual_connection, node._get_child_modules())
            
            
            
    def get_output_nodes(self):
        """
        Gets a list of nodes that have no output edges (including residual connections as output edges)

        :returns: A list of 'output nodes' (the HVN's in the graph that have no children)
        """
        output_nodes = []
        for node in self.nodes:
            if len(node.child_edges) == 0 and node.residual_connection is None:
                output_nodes.append(node)
        return output_nodes



    def get_childless_nodes(self):
        """
        Gets a list of nodes that have no output edges (excluding residual connections as output edges)

        :returns: A list of 'output nodes' (the HVN's in the graph that have no children)
        """
        output_nodes = []
        for node in self.nodes:
            if len(node.child_edges) == 0:
                output_nodes.append(node)
        return output_nodes
    
    
    
    def get_edge_associated_with_module(self, module):
        """
        Gets the HVE associated with some pytorch module 'module'. We return the first edge found, and assume that every 
        module is associated with at most one HVE edge.
        
        :param module: The module to get the associated edge for
        :returns: The HVE object associated with the module given
        """
        for node in self.nodes:
            for edge in node.parent_edges:
                if edge.pytorch_module is module:
                    return edge
        return None



    def concat(self, hvns):
        """
        Concatenates a list of hvn objects, and corresponds to the same operation as

        This "edits" the graph (it creates a new node, that replaces all of the hvn's in 'hvns')

        :param hvns: A list of HVN objects to be concatenated in the HVG
        :returns: The new HVN object in the HVG that's replacing the hvn's in 'hvns'
        """
        # Start with some sanity checks
        for hvn in hvns:
            if len(hvn.child_edges) > 0:
                raise Exception("Can't concatenate volumes that are already fed as input to some other module.")
            if hvn.residual_connection is not None:
                raise Exception("Can't concatenate volumes that already are used in a residual connection.")
            if hasattr(hvn.batch_norm, '__len__'):
                raise Exception("Can't concatenate nodes that mix batch_norm lists and single batch norms... yet")

        is_conv = hvns[0]._is_conv_volume_shape()
        for hvn in hvns:
            if hvn._is_conv_volume_shape() != is_conv:
                raise Exception("Cannot concatinate convolutional volumes and linear volumes.")

        if is_conv:
            _, h, w = hvns[0].hv_shape
            for hvn in hvns:
                if h != hvn.hv_shape[1] or w != hvn.hv_shape[2]:
                    raise Exception("Cannot concatenate convolutional volumes with different spatial dimensions.")

        # Compute the new shape
        new_channels = 0
        for hvn in hvns:
            new_channels += hvn.hv_shape[0]
        new_shape = (new_channels,)
        if is_conv:
            _, h, w = hvns[0].hv_shape
            new_shape = (new_channels, h, w)

        # Concatenate all of the input modules and input hvns
        new_input_modules = []
        new_input_hvns = []
        new_batch_norms = []
        for hvn in hvns:
            new_input_modules.extend(hvn._get_parent_modules())
            new_input_hvns.extend(hvn._get_parent_hvns())
            new_batch_norms.append(hvn.batch_norm)

        # Check that we're not accidentally introducing some loop (could be possible to encorporate, but not now)
        # Also, this doesn't do a proper check for a loop (for more than 2 edges)
        for hvn in hvns:
            if hvn in new_input_hvns:
                raise Exception("Can't concatenate volumes where one is the input to another")

        # Remove all of the HVNs being concatenated from HVNs list, including all of the edges with a child node being
        # one of the HVNs, and replace it with the concatenated volume
        for hvn in self.nodes:
            new_child_edges = []
            for edge in hvn.child_edges:
                if edge.child_node not in hvns:
                    new_child_edges.append(edge)
            hvn.child_edges = new_child_edges
        for hvn in hvns:
            self.nodes.remove(hvn)
        return self.add_hvn(new_shape, input_modules=new_input_modules, input_hvns=new_input_hvns, batch_norm=new_batch_norms)



    def pretty_print(self):
        """
        Pretty prints the graph.
        """
        for i in range(len(self.nodes)):
            s = self.nodes[i]._pretty_print(i)
            s += " -> "
            next = set()
            for edge in self.nodes[i].child_edges:
                j = self.nodes.index(edge.child_node)
                next.add(j)
            next = list(next)
            for j in range(len(next)):
                s += self.nodes[next[j]]._pretty_print(next[j])
                s += ", " if j < len(next) - 1 else ";"
            print(s)



    
    
    
class HVN(object):
    """
    Class representing a node in the hidden volume graph.

    It keeps track of any batch norms that operate on this volume, and any residual connections that it is used in.

    After this hvn we wanted to add another hvn, computed from a nn.Module that doesn't have any parameters. Then we
    can add it into this hvn object as a "pseudo hvn". An example of this is when we want to use a max_pool nn.Module.
    The max_pool nn.Module should be ignored in r_2_wider_r directly, but we still need to take into account the shaping
    changes that it makes. This information can be represented using the pseudo_hvn_shape in this object.

    If this hvn includes a 'pseudo hvn', then any batch norm is associated with *this* hidden volume, whereas any
    residual connection is associated with the *pseudo hvn*.
    """
    def __init__(self, hv_shape, input_modules=[], input_hvns=[], batch_norm=None, residual_connection=None):
        """
        Initialize a node to be used in the hidden volume graph (HVG). It keeps track of a hidden volume shape and any 
        associated batch norm layers, and any associated Residual_Connection objects (which mean that this volume
        is used over a residual connection).
        
        :param hv_shape: The shape of the volume being represented
        :param input_modules: the nn.Modules where input_modules[i] takes input with shape from input_hvns[i] to be 
                concatenated for this hidden volume (node).
        :param input_hvns: A list of HVN objects that are parents/inputs to this HVN
        :param batch_norm: Any batch norm layer that's associated with this hidden volume, None if there is no batch norm.
                In the case where the volume is formed by a concatenation (len(input_hvns) > 1), then, we can provide a
                list
        :param residual_connection: The Residual_Connection object associated with this volume. (I.e. if this volume x is
                used for a residual connection at some point later in the network y, with y = residual_connection(y,x)).
                This should be none if it's not used as part of a residual connection.
        """
        # Keep track of the state
        self.hv_shape = hv_shape
        self.child_edges = []
        self.parent_edges = []
        self.batch_norm = batch_norm
        self.residual_connection = residual_connection
        
        # Make the edges between this node and it's parents/inputs
        self._link_nodes(input_modules, input_hvns)

        # Check for invalid number of batch norm
        if hasattr(batch_norm, '__len__') and len(batch_norm) != len(input_hvns):
            raise Exception("If batch norm is a list of nn.BatchNorm instances, it has to be the same length as the number of inputs to the hvn.")


    
    def _link_nodes(self, input_modules, input_hvns):
        """
        Links a list of hvn's to this hvn, with edges that correspond to PyTorch nn.Modules
        
        :param input_modules: the nn.Modules where input_modules[i] takes input with shape from input_hvns[i] to be 
                concatenated for this hidden volume (node).
        :param input_hvns: A list of HVN objects that are parents/inputs to this HVN
        """ 
        # Check for invalid input
        if len(input_hvns) != len(input_modules):
            raise Exception("When linking nodes in HVG need to provide the same number of input nodes as modules")
        
        # Iterate through all of the inputs, make an edge and update it in the input HVN and this HVN appropriately
        for i in range(len(input_hvns)):
            parent_hvn = input_hvns[i]
            edge = HVE(parent_node=parent_hvn, child_node=self, module=input_modules[i])
            parent_hvn.child_edges.append(edge)
            self.parent_edges.append(edge)



    def _get_parent_hvns(self):
        """
        Gets a list of HVN objects from the parent edges (in the same order)
        """
        return [edge.parent_node for edge in self.parent_edges]
    
    
    
    def _get_parent_modules(self):
        """
        Gets a list of nn.Modules from the parent edges (in the same order)
        """
        return [edge.pytorch_module for edge in self.parent_edges]
    
    
    
    def _get_child_modules(self):
        """
        Gets a list of nn.Modules from the child edges (in the same order)
        """
        return [edge.pytorch_module for edge in self.child_edges]
    
    
    
    def _is_conv_volume_shape(self):
        """
        Volumes output by conv layers are 3D, and linear are 1D.
        """
        return len(self.hv_shape) == 3



    def _is_conv_volume(self):
        """
        Returns a boolean that says if this volume should always be considered a convolutional volume.
        Volumes that aren't always considered a convolutional volume are those that are fed into linear layers, even
        if they are output from a

        Specifically, we define a volume to be a 'conv_volume' iff in the tree rooted at this HVN there is a Conv2d
        module on some edge of the tree (i.e. if there is some computation path using a conv in the "future")

        Assumes that there are no cycles in the HVG (which should be enforced by the HVG).

        :return: If the HVN is considered a 'conv_volume' as defined above
        """
        queue = copy.copy(self.child_edges)
        while len(queue) > 0:
            edge = queue.pop(0)
            if type(edge.pytorch_module) == nn.Conv2d:
                return True
            queue.extend(edge.child_node.child_edges)
        return False



    def _pretty_print(self, index):
        """
        Pretty prints this volume shape, with a tagged index
        :returns: A string for this hvn
        """
        return "({index}, {shape})".format(index=index, shape=self.hv_shape)
            
            
    
    
    
class HVE(object):
    """
    Class representing an edge in the hidden volume graph. It's basically a glorified pair object, with some reference
    to a nn.Module as well
    """
    def __init__(self, parent_node, child_node, module):
        """
        Defines an edge running from the 'parent_node' to the 'child_node' 
        
        :param parent_node: The parent HVN object
        :param child_node: The child HVN object
        :param module: The nn.Module that takes input from the parent node's hidden volume, and produces it's contribution
                to the child node's volume (note that a node may have multiple parents, who's contributions are concatenated)
        """
        self.parent_node = parent_node
        self.child_node = child_node
        self.pytorch_module = module
        
        
        

        
def widen_network_(network, new_channels=0, new_hidden_nodes=0, init_type='match_std', function_preserving=True,
                   multiplicative_widen=True, mfactor=2, net_morph=False):
    """
    We implement a loop that loops through all the layers of a network, according to what we will call the 
    enum_layers interface. The interface should return, for every hidden volume, the layer before it and 
    any batch norm layer associated with it, along with the shape for the current hidden volume.
    
    For simplicity we add 'new_channels' many new channels to hidden volumes, and 'new_units' many additional 
    hidden units. By setting one of them to zero we can easily just widen the convs or fc layers of a network
    independently.

    To be explicit, we will use 'new_channels' in R2WiderR whenever the volume will be (eventually) fed into another
    convolutional layer, otherwise, we go with 'new_hidden_nodes' (probably a lower, more reasonable number of new hidden units)
    
    :param network: The nn.Module to be widened, which must implement the R2WiderR interface.
    :param new_channels: The number of new channels/hidden units to add in conv layers
    :param new_hidden_nodes: The number of new hidden nodes to add in fully connected layers
    :param init_type: The initialization type to use for new variables
    :param function_preserving: If we want the widening to be function preserving
    :param multiplicative_widen: If we want to extend channels by scaling (multiplying num channels) rather than adding
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    :param net_morph: If we should widen according to the net morph widening alghorithm, rather than R2WiderR.
    :return: A reference to the widened network
    """
    # Create the hidden volume graph
    hvg = network.hvg()
        
    # Iterate through the hvg, widening appropriately in each place
    prev_channels_or_nodes_to_add = -1
    for prev_layers, shape, is_conv_volume, batch_norm, residual_connection, next_layers in hvg.node_iterator():
        channels_or_nodes_to_add = new_channels if is_conv_volume else new_hidden_nodes

        # Sanity check that we haven't made an oopsie with the widening operation with pooling layers
        if channels_or_nodes_to_add != prev_channels_or_nodes_to_add:
            for layer in prev_layers:
                if type(layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d] and hasattr(layer, 'r2r_cache'):
                    raise Exception("Arguments to widen_network_ caused a pooling layer to be 'widened' as part of "
                                    "next_layers in R2WiderR, but, when widening as part of prev_layers using use an "
                                    "inconsistent number of channels.")
        prev_channels_or_nodes_to_add = channels_or_nodes_to_add

        # Perform the widening
        if channels_or_nodes_to_add == 0:
            continue
        r_2_wider_r_(prev_layers, shape, next_layers, batch_norm, residual_connection, channels_or_nodes_to_add,
                     init_type, function_preserving, multiplicative_widen, mfactor, net_morph)
        
    # Return model for if someone want to use this in an assignment form etc
    return network
        


        
        
"""
Deepening networks.
"""





class Deepened_Network(nn.Module):
    """
    A heper class that encapsulates a "base" network and allows layers to be "inserted" into the middle of the 
    base network.
    
    The base network must implement the R2DeeperR interface, and must additionally implement the R2R interface if
    we wish to be further widened.

    And modules used in the deepening must implement the R2DeeperRBlock interface.
    """
    def __init__(self, base_network):
        """
        :param base_network: The base network that is to be "deepened".
        """
        super(Deepened_Network, self).__init__()
        
        if type(base_network) is Deepened_Network:
            raise Exception("We shouldn't be recursively create Deepened_Network instances, it should instead be extended.")
        
        self.base_network = base_network
        self.conv_extensions = []
        self.fc_extensions = []
        
        
    def forward(self, x):
        """
        The forward pass of the network, whcih basically interleves the new extensions with the base networks forward pass
        :param x: The input to the network
        :return: The output from the deepened network
        """
        x = self.base_network.conv_forward(x)
        for conv_module in self.conv_extensions: 
            x = conv_module(x)
        x = self.base_network.fc_forward(x)
        for fc_module in self.fc_extensions:
            x = fc_module(x)
            x = F.relu(x)
        return self.base_network.out_forward(x)
        
        
    def deepen(self, module):
        """
        Extends the network with nn.Module 'module'. We assume that the layer is to be applied between the convolutional 
        stack and the fc stack, unless 'module' is of type nn.Linear.
        
        To register the parameters for autograd computation, we need to use "add_module", which is similar to 
        "register_parameter".
        
        Note: we take the liberty to assume that the shapings of layer are correct to just work.

        :param module: The nn.Module to deepen the network with. Must implemnt the R2DeeperRBlock interface.
        """
        if type(module) is nn.Linear:
            self.fc_extensions.append(module)
            self.add_module("fc_ext_{n}".format(n=len(self.fc_extensions)), module)
        else:
            self.conv_extensions.append(module)
            self.add_module("conv_ext_{n}".format(n=len(self.conv_extensions)), module)
            
            
            
    def hvg(self):
        """
        Implements the hvg interface, assuming that all components of the deepened network do too (base network and new 
        modules).

        Assumes that the base network implements the R2R interface (i.e. it has been deepened, and we wish to
        widen it, if we are calling hvg after all).

        It also assumes that all of the layers 'module' in self.conv_extensions implement the R2DeeperRBlock interface.
        
        This function builds a hidden volume graph for the deepened network.

        :returns: The HVG object defined over this network.
        """
        hvg = HVG(self.base_network.input_shape())
        hvg = self.base_network.conv_hvg(hvg)
        for conv_module in self.conv_extensions:
            hvg = conv_module.conv_hvg(hvg)
        hvg = self.base_network.fc_hvg(hvg)
        for fc_layer in self.fc_extensions:
            hvg.add_hvn((fc_layer.out_features,) [fc_layer])
        return hvg

        
        
        
        
def make_deeper_network_(network, layer):
    """
    Given a network 'network', create a deeper network adding in a new layer 'layer'. 
    
    We assume the our network is build into a conv stack, which feeds into a fully connected stack.

    We also make sure that if the network was already on the GPU, then the new network is also on the GPU.

    :param network: The nn.Module to be deepened, that implements the R2DeeperR interface.
    :param layer: A nn.Module to be used in the deepening, that implements the R2DeeperRBlock interface.
    :returns: A nn.Module for the deepened network (this may be new object, but still possibly clobbers 'network' input)
    """
    if type(network) is not Deepened_Network:
        network = Deepened_Network(network)
        if next(network.parameters()).is_cuda:
            network = cudafy(network)
    network.deepen(layer)
    return network


