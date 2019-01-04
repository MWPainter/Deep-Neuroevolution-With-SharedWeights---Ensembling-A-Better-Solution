import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

import math
from collections import defaultdict

from r2r.init_utils import *
from r2r.module_utils import *
from r2r.r2r import HVN, HVG, HVE, Deepened_Network





# Only export things that actually widen/deepen volumes, and not helper functions
all = ['net_2_wider_net_',
       'net2net_widen_network_',
       'net2net_make_deeper_network_'
       'Net2Net_conv_identity']  # make_deeper_network = r2deeperr if add identity initialized module

"""
Widening hidden volumes
"""





def _round_up_multiply(a, b, m):
    """
    Performs a*b and rounds up to the nearest m. Note that (x+m-1)//m, is a divide by m rounded up
    """
    prod = int(math.ceil(((a*b + m - 1) // m) * m))
    return prod





def net_2_wider_net_(prev_layers, next_layers, volume_shape, batch_norm,
                     extra_channels=0, multiplicative_widen=True, add_noise=True, mfactor=2):
    """
    Single interface for r2widerr transforms, where prev_layers and next_layers could be a single layer.
    
    
    This function will widen the hidden volume output by a convolutional layer, or the hidden units output 
    by a linear layer. 'prev_layer' refers to the layer who's output we are widening, and, we also take 
    'next_layer' which we need to make compatable with the new wider input it's going to recieve.
    
    For now, with convolutions, we only support variable sizes kernels, strides and padding. We don't support 
    any non-default padding and so on.
    
    Finally, we allow ourselves to initialize the new varialbles in such a way that the functions output is 
    preserved. This is implemented using the 'alpha' variable of the underlying functions that initialize the 
    new weights.
    
    For now we support the following *ordered* pairs of nn.Modules:
    (nn.Conv2d, nn.Conv2d)
    (nn.Conv2d, nn.Linear)
    (nn.Linear, nn.Linear)
    
    :param prev_layers: A (list of) layer(s) before the hidden volume/units being widened (their outputs are 
            concatenated together)
    :param next_layers: A (list of) layer(s) layer after the hidden volume/units being widened (inputs are the 
            concatenated out)
    :param volume_shape: 
    :param batch_norm: The batch norm function associated with the volume being widened (None if there is no batch norm)
    :param input_shape: The shape of the input
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param function_preserving: If we wish for the widening to preserve the function I/O
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    """
    # Handle inputting of single input/output layers
    if type(prev_layers) != list:
        prev_layers = [prev_layers]
    if type(next_layers) != list:
        next_layers = [next_layers]

    _net_2_wider_net_(prev_layers, next_layers, volume_shape, batch_norm, extra_channels,
                      multiplicative_widen, add_noise, mfactor)





def _net_2_wider_net_(prev_layers, next_layers, volume_shape, batch_norm, extra_channels,
                      multiplicative_widen, add_noise, mfactor):
    """  
    The full internal implementation of net_2_wider_net_. See description of net_2_wider_net_.
    More helper functions are used.
    """
    # For us to be able to perform a function preserving transforms, extra channels must be even and non-negative
    if extra_channels <= 0:
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
        if type(prev_layers[i]) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d] and not hasattr(prev_layers[i], 'r2r_volume_slice_indices'):
            raise Exception("AvgPool2d or MaxPool2d in prev_layers when not used previously in a R2WiderR call in "
                            "next_layers, which is necessary to create a cache of values needed for the transformation.")

    # Check that the volume shape is either linear or convolutional
    if len(volume_shape) not in [1, 3]:
        raise Exception("Volume shape must be 1D or 3D for R2WiderR to work")

    # Get if we have a linear input or not
    input_is_linear = any([type(prev_layer) == nn.Linear for prev_layer in prev_layers])

    #  Work out the number of hiden units per new channel
    # (for fc pretend 1x1 spatial resolution, so 1 per channel) (for conv this is width*height)
    # For conv -> linear layers this can be a little complex. But we always know the number of channels from the prev kernels
    channels_in_volume = np.sum([_get_output_channels_from_layer(layer) for layer in prev_layers])
    total_hidden_units = np.prod(volume_shape)
    new_hidden_units_in_next_layer_per_new_channel = total_hidden_units // channels_in_volume


    # Sanity check
    if input_is_linear and new_hidden_units_in_next_layer_per_new_channel != 1:
        raise Exception("Number of 'new hidden_units per new channel' must be 1 for linear. Something went wrong :(.")

    # Compute the slicing of the volume from the input (to widen outputs appropraitely) (and the new slices for the output)
    module_slices_indices = [0]
    widened_module_slices_indices = [0]
    for prev_layer in prev_layers:
        # Get slices output from a module (this will be [0,out_channels] for any nn.Conv2d or nn.Linear (not
        # necessarily for pooling, as this is used to propogate mappings))
        local_slice_indices = _compute_new_volume_slices_from_layer(0, prev_layer)

        # Compute the next module slice (for current volume)
        base_index = module_slices_indices[-1]
        module_slices_indices.append(base_index + local_slice_indices[-1])

        # Compute next module slice (for widened volume)
        widened_base_index = widened_module_slices_indices[-1]
        channels_to_add_for_module = len(local_slice_indices) * extra_channels
        if multiplicative_widen:
            channels_to_add_for_module = _round_up_multiply(local_slice_indices[-1], extra_channels-1, mfactor)
        widened_module_slices_indices.append(widened_base_index + local_slice_indices[-1] + channels_to_add_for_module)

    # Compute a mapping between old channels of the volume and new channels of the widened volume
    local_channel_maps = []
    channel_maps = []
    base_index = 0
    for prev_layer in prev_layers:
        out_channels = _get_output_channels_from_layer(prev_layer)

        if type(prev_layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
            if not hasattr(prev_layer, 'r2r_channel_map_cache'):
                raise Exception("Pooling layer used in prev layers before it was called as part of next layers in net2net")
            local_channel_map = prev_layer.r2r_channel_map_cache

        else:
            identity_local_map = np.arange(out_channels)

            map_extra_channels = extra_channels
            if multiplicative_widen:
                map_extra_channels = _round_up_multiply(out_channels, extra_channels - 1, mfactor)

            extra_local_channels_map = np.random.randint(out_channels, size=map_extra_channels)
            local_channel_map = np.concatenate([identity_local_map, extra_local_channels_map], axis=0)

        local_channel_maps.append(local_channel_map)
        partial_channel_map = local_channel_map + base_index
        channel_maps.append(partial_channel_map)

        base_index += out_channels

    channel_map = np.concatenate(channel_maps, axis=0)


    # Iterate through all of the prev layers, and widen them appropraitely
    input_is_linear = False
    for (layer_index, prev_layer) in enumerate(prev_layers):
        input_is_linear = input_is_linear or type(prev_layer) is nn.Linear
        _net2net_widen_output_channels_(prev_layer, local_channel_maps[layer_index], input_is_linear, add_noise)

    # Widen batch norm appropriately 
    if batch_norm:
        _net2net_extend_bn_(batch_norm, channel_map, module_slices_indices, widened_module_slices_indices)

    # Iterate through all of the next layers, and widen them appropriately. (Needs the slicing information to deal with concat)
    for (layer_index, next_layer) in enumerate(next_layers):
        _net2net_widen_input_channels_(next_layer, channel_map, new_hidden_units_in_next_layer_per_new_channel,
                                       input_is_linear, module_slices_indices)





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





def _net2net_extend_bn_(bn, channel_map, module_slices_indices, widened_module_slices_indices):
    """
    Extend batch norm with new_channels_per_slice many extra units.
    Replicate the channels indicated by extra_channels_mappings.

    :param bn: The batch norm layer to be extended
    :param channel_map: The mapping between layers in the unwidened volume to the widened volume
    :param module_slices_indices: The indices to be able to slice the volume according to which input module was used
    :param widened_module_slices_indices: The indices to be able to slice the volume according to which input module
            was used (after widening has occurred)
    """
    # Sanity checks
    bn_is_array = hasattr(bn, '__len__')
    if hasattr(bn, '__len__') and len(bn) + 1 != len(module_slices_indices):
        raise Exception("Number of batch norms and volume slice indices inconsistent.")

    if not bn_is_array:
        old_scale = bn.weight.data.cpu().numpy()
        old_shift = bn.bias.data.cpu().numpy()
        old_running_mean = bn.running_mean.data.cpu().numpy()
        old_running_var = bn.running_var.data.cpu().numpy()
    else:
        total_channels = module_slices_indices[-1]
        old_scale = np.ones(total_channels).astype(np.float32)
        old_shift = np.zeros(total_channels).astype(np.float32)
        old_running_mean = np.zeros(total_channels).astype(np.float32)
        old_running_var = np.ones(total_channels).astype(np.float32)

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
    new_scale = old_scale[channel_map]
    new_shift = old_shift[channel_map]
    new_running_mean = old_running_mean[channel_map]
    new_running_var = old_running_var[channel_map]

    # Assign to batch norm(s)
    if not bn_is_array:
        _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)

    else:
        for i in range(len(bn)):
            if bn[i] is None:
                continue
            beg = widened_module_slices_indices[i]
            end = widened_module_slices_indices[i+1]

            new_scale_slice = new_scale[beg:end]
            new_shift_slice = new_shift[beg:end]
            new_running_mean_slice = new_running_mean[beg:end]
            new_running_var_slice = new_running_var[beg:end]

            _assign_to_batch_norm_(bn[i], new_scale_slice, new_shift_slice, new_running_mean_slice,
                                   new_running_var_slice)





def _net2net_widen_output_channels_(prev_layer, channel_map, input_is_linear, noise=True):
    """
    Helper function for net2widernet. Containing all of the logic for widening the output channels of the 'prev_layers'.
    
    :param prev_layer: A layer before the hidden volume/units being widened
    :param channel_map: Mapping of channels to copy for output channels in prev_layer
    :param input_is_linear: If any of the layers are linear
    :param noise: If we should add noise
    """
    # Logic for conv2d and linear is the same in this case, just use smart indexing on output layers
    if type(prev_layer) in [nn.Conv2d, nn.Linear]:
        # Get the new weight (kernel/matrix)
        prev_weight = prev_layer.weight.data.cpu().numpy()
        new_weight = prev_weight[channel_map]

        # Compute the new bias if there is any
        new_bias = None
        if prev_layer.bias is not None:
            prev_bias = prev_layer.bias.data.cpu().numpy()
            new_bias = prev_bias[channel_map]

        # Add noise if we wish
        if noise:
            noise = np.random.normal(scale=5e-5 * np.std(new_weight),
                                     size=new_weight.shape)
            new_weight += noise

        # Make assignment to module
        if type(prev_layer) is nn.Conv2d:
            _assign_kernel_and_bias_to_conv_(prev_layer, new_weight, new_bias)
        else:
            _assign_weights_and_bias_to_linear_(prev_layer, new_weight, new_bias)

    # Pooling layers need to clean up caches
    elif type(prev_layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
        # Unpack, and check consistency
        prev_input_is_linear = prev_layer.r2r_prev_input_is_linear
        if prev_input_is_linear != input_is_linear:
            raise Exception("Pooling layer allows accidental mixing of conv and linear layers at the next layer.")

        # Delete the values that we stored in the nn.Module as a cache
        del prev_layer.r2r_prev_input_is_linear
        del prev_layer.r2r_channel_map_cache
        del prev_layer.r2r_old_channels_propagated
        del prev_layer.r2r_volume_slice_indices





def _net2net_widen_input_channels_(next_layer, channel_map, new_hidden_units_in_next_layer_per_new_channel,
                                   input_is_linear, module_slices_indices):
    """
    Helper function for net2widernet. Containing all of the logic for widening the input channels of the 'next_layers'.
    
    :param next_layer: A layer after the hidden volume/units being widened
    :param channel_map: The mapping between layers in the unwidened volume to the widened volume
    :param new_hidden_units_in_next_layer_per_new_channel: The number of hidden units output per channel from the prev layers
    :param input_is_linear: If the input to these modules comes from a linear layer
    :param module_slices_indices: Indicies into the hidden volume (according to the modules that output them).
    """
    # Check that we don't do linear -> conv, as haven't worked this out yet
    if input_is_linear and type(next_layer) is nn.Conv2d:
        raise Exception("We currently don't handle the nn.Linear -> nn.Conv2d case in r_2_wider_r.")

    if type(next_layer) in [nn.Conv2d, nn.Linear]:
        # Get the old weight (kernel/matrix)
        old_weight = next_layer.weight.data.cpu().numpy()
        out_channels = old_weight.shape[0]

        # If next layer is linear, convert the weight shape from (n_out, n_in) to (n_out, n_in/units_per_channel, units_per_channel, 1)
        if type(next_layer) is nn.Linear:
            old_weight = np.reshape(old_weight, (out_channels, -1, new_hidden_units_in_next_layer_per_new_channel, 1))

        # Compute divisors (reshaping for unambiguity with broadcasting)
        counts = defaultdict(float)
        for i in range(len(channel_map)):
            counts[channel_map[i]] += 1.0
        divisors = []
        for i in range(len(channel_map)):
            divisors.append(counts[channel_map[i]])
        divisors = np.reshape(np.array(divisors).astype(np.float32), (1, -1, 1, 1))

        # Duplicate input layers and divide to get new weights
        new_weight = old_weight[:, channel_map] / divisors

        # Make assignment to module (remembering to reshape the linear layer back)
        if type(next_layer) is nn.Conv2d:
            _assign_kernel_and_bias_to_conv_(next_layer, new_weight)
        else:
            new_weight = np.reshape(new_weight, (out_channels, -1))
            _assign_weights_and_bias_to_linear_(next_layer, new_weight)

    # Pooling layers need to store stuff in caches (to be used when they're in prev_layer)
    elif type(next_layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d]:
        next_layer.r2r_prev_input_is_linear = input_is_linear
        next_layer.r2r_channel_map_cache = channel_map
        next_layer.r2r_old_channels_propagated = module_slices_indices[-1]
        next_layer.r2r_volume_slice_indices = module_slices_indices

    else:
        raise Exception("We can only handle output nn.Modules that are Linear or Conv2d at the moment.")





"""
Widening entire networks (by building graphs over the networks)
"""





def net2net_widen_network_(network, new_channels=0, new_hidden_nodes=0, multiplicative_widen=True, add_noise=True,
                           mfactor=2):
    """
    We implement a loop that loops through all the layers of a network, according to what we will call the
    enum_layers interface. The interface should return, for every hidden volume, the layer before it and
    any batch norm layer associated with it, along with the shape for the current hidden volume.

    For simplicity we add 'new_channels' many new channels to hidden volumes, and 'new_units' many additional
    hidden units. By setting one of them to zero we can easily just widen the convs or fc layers of a network independently.

    :param network: The nn.Module to be widened
    :param new_channels: The number of new channels/hidden units to add in conv layers
    :param new_hidden_nodes: The number of new hidden nodes to add in fully connected layers
    :param init_type: The initialization type to use for new variables
    :param function_preserving: If we want the widening to be function preserving
    :param multiplicative_widen: If we want to extend channels by scaling (multiplying num channels) rather than adding
    :param mfactor: When adding say 1.4 times the channels, we round up the number of new channels to be a multiple of
            'mfactor'. This parameter has no effect if multiplicative_widen == False.
    :return: A reference to the widened network
    """
    #  Create the hidden volume graph
    hvg = network.hvg()

    # Iterate through the hvg, widening appropriately in each place
    for prev_layers, shape, is_conv_volume, batch_norm, _, next_layers in hvg.node_iterator():
        channels_or_nodes_to_add = new_channels if is_conv_volume else new_hidden_nodes

        # Sanity check that we haven't made an oopsie with the widening operation with pooling layers
        for layer in prev_layers:
            if type(layer) in [nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d] and not hasattr(layer, 'r2r_channel_map_cache'):
                raise Exception("Arguments to widen_network_ caused a pooling layer to be 'widened' as part of "
                                "next_layers in R2WiderR, but, when widening as part of prev_layers using use an "
                                "inconsistent number of channels.")

        # Perform the widening
        if channels_or_nodes_to_add == 0:
            continue
        net_2_wider_net_(prev_layers, next_layers, shape, batch_norm, extra_channels=channels_or_nodes_to_add,
                         multiplicative_widen=multiplicative_widen, add_noise=add_noise, mfactor=mfactor)

    # Return model for if someone want to use this in an assignment form etc
    return network





"""
Deepening networks.
"""





def net2net_make_deeper_network_(network, layer, batch):
    """
    Given a network 'network', create a deeper network adding in a new layer 'layer'.

    We assume the our network is build into a conv stack, which feeds into a fully connected stack.
    """
    if type(network) is not Deepened_Network:
        network = Deepened_Network(network)
    network.deepen(layer)

    for bn_layer in layer.bn_ble():
        momentum = bn_layer.momentum
        bn_layer.reset_running_stats()
        bn_layer.momentum = 1.0
        network(batch)
        new_weights = np.sqrt(bn_layer.running_var.numpy() + bn_layer.eps)
        _assign_to_batch_norm_(bn_layer, new_weights, bn_layer.running_mean.numpy(),
                               bn_layer.running_mean.numpy(), bn_layer.running_var.numpy())
        bn_layer.momentum = momentum

    return network





class Net2Net_ResBlock_identity(nn.Module):

    def __init__(self, input_channels, intermediate_channels, output_channels, identity_initialize=True,
                 input_spatial_shape=None, input_volume_slices_indices=None):
        """
        Initialize the filters, optionally making this identity initialized.
        All convolutional filters have the same number of output channels
        """
        # Superclass initializer
        super(Net2Net_ResBlock_identity, self).__init__()

        self.input_channel = input_channels

        # Check that we gave the correct number of intermediate channels
        if len(intermediate_channels) != 3:
            raise Exception("Need to specify 3 intemediate channels in the resblock")

        # Stuff that we need to remember
        self.input_spatial_shape = input_spatial_shape
        self.intermediate_channels = intermediate_channels
        self.output_channels = output_channels

        # Actual nn.Modules for the network
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=intermediate_channels[0], momentum=1)
        self.update_conv(self.conv1)

        self.conv2 = nn.Conv2d(intermediate_channels[0], intermediate_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=intermediate_channels[1], momentum=1)
        self.update_conv(self.conv2)

        self.conv3 = nn.Conv2d(intermediate_channels[1], intermediate_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=intermediate_channels[2], momentum=1)
        self.update_conv(self.conv3)

        self.conv4 = nn.Conv2d(intermediate_channels[2], output_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=output_channels, momentum=1)
        self.update_conv(self.conv4)

        self.relu = F.relu



    def update_conv(self, conv_layer, noise=False):

        conv_kernel = conv_layer.weight.data.cpu()
        conv_bias = conv_layer.bias.data.cpu()

        out_channels, in_channels, height, width = conv_kernel.shape

        conv_kernel.zero_()
        conv_bias.zero_()

        center_height = (height - 1) // 2
        center_width = (width - 1) // 2

        for i in range(0, out_channels):
            conv_kernel.narrow(0, i, 1).narrow(1, i, 1).narrow(2, center_height, 1).narrow(3, center_width, 1).fill_(1)

        if noise:
            noise = np.random.normal(scale=5e-5, size=list(conv_kernel.size()))
            conv_kernel += t.FloatTensor(noise).type_as(conv_kernel)

        _assign_kernel_and_bias_to_conv_(conv_layer, conv_kernel.numpy(), conv_bias.numpy())



    def forward(self, x):
        """
        Forward pass through this residual block

        :param x: the input
        :return: THe output of applying this residual block to the input
        """
        # Forward pass through residual part of the network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        out = x
        return out



    def bn_ble(self):
        yield (self.bn1)
        yield (self.bn2)
        yield (self.bn3)
        yield (self.bn4)



    def conv_hvg(self, cur_hvg):
        """
        Extends a hidden volume graph 'hvg'.

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

        # First hidden
        cur_hvg.add_hvn((self.conv1.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]),
                        input_modules=[self.conv1], batch_norm=self.bn1)

        # Second hidden volume
        cur_hvg.add_hvn((self.conv2.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]),
                        input_modules=[self.conv2], batch_norm=self.bn2)

        # Third hidden volume
        cur_hvg.add_hvn((self.conv3.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]),
                        input_modules=[self.conv3], batch_norm=self.bn3)

        # Fourth (output) hidden volume (second of r2r block)
        cur_hvg.add_hvn((self.conv4.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]),
                        input_modules=[self.conv4], batch_norm=self.bn4)

        return cur_hvg



    def hvg(self):
        raise Exception("hvg() not implemented directly for net2net conv identity block.")





class Net2Net_conv_identity(nn.Module):

    def __init__(self, input_channels, kernel_size, activation_function='ReLU',
                 input_spatial_shape=None):
        """
        Initialize the filters, optionally making this identity initialized.
        All convolutional filters have the same number of output channels
        """
        # Superclass initializer
        super(Net2Net_conv_identity, self).__init__()

        self.input_channel = input_channels
        self.kernel_size = kernel_size

        # Stuff that we need to remember
        self.input_spatial_shape = input_spatial_shape

        assert kernel_size[0] % 2 == 1, "Kernel size needs to be odd"
        assert kernel_size[1] % 2 == 1, "Kernel size needs to be odd"
        pad_h = int((kernel_size[0] - 1) / 2)

        # Actual nn.Modules for the network
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, padding=pad_h)
        self.bn = nn.BatchNorm2d(num_features=input_channels, momentum=1)
        self.relu = F.relu

        self.update_conv(self.conv)



    def update_conv(self, conv_layer, noise=False):

        conv_kernel = conv_layer.weight.data.cpu()
        conv_bias = conv_layer.bias.data.cpu()

        out_channels, in_channels, height, width = conv_kernel.shape

        conv_kernel.zero_()
        conv_bias.zero_()

        center_height = (height - 1) // 2
        center_width = (width - 1) // 2

        for i in range(0, in_channels):
            conv_kernel.narrow(0, i, 1).narrow(1, i, 1).narrow(2, center_height, 1).narrow(3, center_width, 1).fill_(1)

        if noise:
            noise = np.random.normal(scale=5e-5, size=list(conv_kernel.size()))
            conv_kernel += t.FloatTensor(noise).type_as(conv_kernel)

        _assign_kernel_and_bias_to_conv_(conv_layer, conv_kernel.numpy(), conv_bias.numpy())



    def forward(self, x):
        """
        Forward pass through this residual block

        :param x: the input
        :return: THe output of applying this residual block to the input
        """
        # Forward pass through residual part of the network

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        out = x

        return out



    def bn_ble(self):
        yield (self.bn)



    def conv_hvg(self, cur_hvg):
        """
        Extends a hidden volume graph 'hvg'.
        :param cur_hvg: The HVG object of some larger network (that this resblock is part of)
        :param input_nodes: The node that this module takes as input
        :return: The hvn for the output from the resblock
        """
        # First hidden
        cur_hvg.add_hvn((self.conv.weight.data.size(0), self.input_spatial_shape[0], self.input_spatial_shape[1]),
                        input_modules=[self.conv], batch_norm=self.bn)
        return cur_hvg



    def hvg(self):
        raise Exception("hvg() not implemented directly for net2net conv identity block.")
