import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from r2r.init_utils import *
from r2r.module_utils import *

###############################################
##### Net2Net implementation adapted from #####
##### https://github.com/erogol/Net2Net   #####
###############################################


# Only export things that actually widen/deepen volumes, and not helper functions
all = ['net_2_wider_net_',
       'HVG',
       'HVN',
       'HVE',
       'net2net_widen_network_',
       'net2net_make_deeper_network_'
       'Net2Net_conv_identity'
       'Deepened_Network']  # make_deeper_network = r2deeperr if add identity initialized module

"""
Widening hidden volumes
"""


def net_2_wider_net_(prev_layers, next_layers, volume_shape, batch_norm,
                     extra_channels=0, scaled=True):
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
    """
    # Handle inputting of single input/output layers
    if type(prev_layers) != list:
        prev_layers = [prev_layers]
    if type(next_layers) != list:
        next_layers = [next_layers]

    _net_2_wider_net_(prev_layers, next_layers, volume_shape, batch_norm, extra_channels, scaled)


def _net_2_wider_net_(prev_layers, next_layers, volume_shape, batch_norm, extra_channels, scaled):
    """  
    The full internal implementation of net_2_wider_net_. See description of net_2_wider_net_.
    More helper functions are used.
    """
    # For us to be able to perform a function preserving transforms, extra channels must be even and non-negative
    if extra_channels <= 0:
        raise Exception("Invalid number of extra channels in widen.")

    # Check that the types of all the in/out layers are the same
    for i in range(1, len(prev_layers)):
        if type(prev_layers[0]) != type(prev_layers[i]):
            raise Exception("All input layers in R2WiderR need to be the same type, nn.Conv2D or nn.Linear")
    for i in range(1, len(next_layers)):
        if type(prev_layers[0]) != type(next_layers[i]):
            raise Exception("All output layers in R2WiderR need to be the same type, nn.Conv2D or nn.Linear")

    #  Work out the number of params per new channel (for fc this is 1)
    new_params_per_new_channel = 1
    if len(volume_shape) == 3:
        _, height, width = volume_shape
        new_params_per_new_channel = height * width

    # Check that the volume shape is either linear or convolutional
    if len(volume_shape) not in [1, 3]:
        raise Exception("Volume shape must be 1D or 3D for R2WiderR to work")

    # Get if we have a linear input or not
    is_linear_input = type(prev_layers[0]) is nn.Linear

    #  Work out the number of hiden units per new channel
    # (for fc pretend 1x1 spatial resolution, so 1 per channel) (for conv this is width*height)
    # For conv -> linear layers this can be a little complex. But we always know the number of channels from the prev kernels


    channels_in_volume = np.sum([layer.weight.size(0) for layer in prev_layers])
    total_hidden_units = np.prod(volume_shape)
    new_hidden_units_per_new_channel = total_hidden_units // channels_in_volume

    # Sanity check
    if is_linear_input and new_hidden_units_per_new_channel != 1:
        raise Exception("Number of 'new hidden_units per new channel' must be 1 for linear. Something went wrong :(.")

    # Compute the slicing of the volume from the input (to widen outputs appropraitely)
    volume_slices_indxs = [0]
    for prev_layer in prev_layers:
        new_slice_indx = volume_slices_indxs[-1] + prev_layer.weight.size(0)
        volume_slices_indxs.append(new_slice_indx)

    extra_channels_mappings = dict()
    for (layer_index, prev_layer) in enumerate(prev_layers):
        # Generate a mapping function for each layer, that indicates which original layer is copied in the extra layer.

        mapping = dict()
        out_channels = prev_layer.weight.size(0)

        if scaled:
            map_extra_channels = out_channels * (
                        extra_channels - 1)  # to triple number of channels, *add* 2x the current num

        for extra_channel_index in range(out_channels, out_channels + map_extra_channels):
            original_channel_index = np.random.randint(0, out_channels)
            mapping[extra_channel_index] = original_channel_index

            extra_channels_mappings[layer_index] = mapping

    # Iterate through all of the prev layers, and widen them appropraitely
    input_is_linear = False
    for (layer_index, prev_layer) in enumerate(prev_layers):
        input_is_linear = input_is_linear or type(prev_layer) is nn.Linear
        _net2net_widen_output_channels_(prev_layer, extra_channels, extra_channels_mappings[layer_index], scaled)

    # Widen batch norm appropriately 
    if batch_norm:
        _net2net_extend_bn_(batch_norm, extra_channels, volume_slices_indxs, extra_channels_mappings, scaled)

    # Iterate through all of the next layers, and widen them appropriately. (Needs the slicing information to deal with concat)
    for (layer_index, next_layer) in enumerate(next_layers):
        _net2net_widen_input_channels_(next_layer, extra_channels, volume_slices_indxs, input_is_linear,
                                       new_hidden_units_per_new_channel,
                                       extra_channels_mappings[layer_index], scaled)


def _net2net_extend_bn_(bn, new_channels_per_slice, volume_slice_indxs, extra_channels_mappings, scaled):
    """
    Extend batch norm with new_channels_per_slice many extra units.
    Replicate the channels indicated by extra_channels_mappings.

    :param bn: The batch norm layer to be extended
    :param new_channels_per_slice: The number of new channels to add, per slice
    :param volume_slice_indx: The indices to be able to slice the volume (so that we can add new
    :param scaled: If true, then we should interpret "extra channels" as a scaling factor for the number of outputs
    """
    new_scale_slices = []
    new_shift_slices = []
    new_running_mean_slices = []
    new_running_var_slices = []

    old_scale = bn.weight.data.cpu()
    old_shift = bn.bias.data.cpu()
    old_running_mean = bn.running_mean.data.cpu()
    old_running_var = bn.running_var.data.cpu()

    for i in range(1, len(volume_slice_indxs)):
        beg = volume_slice_indxs[i - 1]
        end = volume_slice_indxs[i]
        extra_channels_mapping = extra_channels_mappings[i - 1]

        # to triple number of channels, *add* 2x the current num
        new_channels = new_channels_per_slice if not scaled else (new_channels_per_slice - 1) * (end - beg)

        out_channels = end - beg
        new_width = out_channels + new_channels

        new_scale = old_scale[beg:end].clone().resize_(new_width)
        new_shift = old_shift[beg:end].clone().resize_(new_width)
        new_running_mean = old_running_mean[beg:end].clone().resize_(new_width)
        new_running_var = old_running_var[beg:end].clone().resize_(new_width)

        new_scale.narrow(0, 0, out_channels).copy_(old_scale[beg:end])
        new_shift.narrow(0, 0, out_channels).copy_(old_shift[beg:end])
        new_running_var.narrow(0, 0, out_channels).copy_(old_running_var[beg:end])
        new_running_mean.narrow(0, 0, out_channels).copy_(old_running_mean[beg:end])

        for index in range(out_channels, new_width):
            new_scale[index] = old_scale[beg:end][extra_channels_mapping[index]]
            new_shift[index] = old_shift[beg:end][extra_channels_mapping[index]]
            new_running_mean[index] = old_running_var[beg:end][extra_channels_mapping[index]]
            new_running_var[index] = old_running_mean[beg:end][extra_channels_mapping[index]]

        new_scale_slices.append(new_scale)
        new_shift_slices.append(new_shift)
        new_running_mean_slices.append(new_running_mean)
        new_running_var_slices.append(new_running_var)

    new_scale = np.concatenate(new_scale_slices)
    new_shift = np.concatenate(new_shift_slices)
    new_running_mean = np.concatenate(new_running_mean_slices)
    new_running_var = np.concatenate(new_running_var_slices)

    _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)


def _net2net_widen_output_channels_(prev_layer, extra_channels, extra_channels_mapping, scaled, noise=True):
    """
    Helper function for net2widernet. Containing all of the logic for widening the output channels of the 'prev_layers'.
    
    :param prev_layer: A layer before the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    """
    if type(prev_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        prev_kernel = prev_layer.weight.data.cpu()
        prev_bias = prev_layer.bias.data.cpu()

        # new conv kernel
        out_channels, in_channels, width, height = prev_kernel.shape
        layer_extra_channels = extra_channels

        if scaled:
            layer_extra_channels = out_channels * (extra_channels - 1) # to triple number of channels, *add* 2x the current num


        new_kernel = prev_kernel.clone()
        new_kernel.resize_(out_channels + layer_extra_channels, in_channels, width, height)

        new_bias = prev_bias.clone()
        new_bias.resize_(out_channels + layer_extra_channels)

        new_kernel.narrow(0, 0, out_channels).copy_(prev_kernel)
        new_bias.narrow(0, 0, out_channels).copy_(prev_bias)

        for index in range(out_channels, out_channels + layer_extra_channels):
            new_kernel.select(0, index).copy_(prev_kernel.select(0, extra_channels_mapping[index]).clone())
            new_bias[index] = prev_bias[extra_channels_mapping[index]]

        if noise:
            noise = np.random.normal(scale=5e-5 * new_kernel.std(),
                                     size=list(new_kernel.size()))
            new_kernel += t.FloatTensor(noise).type_as(new_kernel)

        # assign new conv and bias
        _assign_kernel_and_bias_to_conv_(prev_layer, new_kernel.numpy(), new_bias.numpy())

    elif type(prev_layer) is nn.Linear:
        # unpack linear params to numpy tensors
        prev_matrix = prev_layer.weight.data.cpu()
        prev_bias = prev_layer.bias.data.cpu()

        # new linear matrix
        n_out, n_in = prev_matrix.shape
        layer_extra_channels = extra_channels

        if scaled:
            layer_extra_channels = n_out * (extra_channels - 1) # to triple number of channels, *add* 2x the current num

        new_matrix = prev_matrix.clone()
        new_matrix.resize_(n_out + layer_extra_channels, n_in)

        new_bias = prev_bias.clone()
        new_bias.resize_(n_out + layer_extra_channels)


        new_matrix.narrow(0, 0, n_out).copy_(prev_matrix)
        new_bias.narrow(0, 0, n_out).copy_(prev_bias)

        for index in range(n_out, n_out + layer_extra_channels):
            new_matrix.select(0, index).copy_(prev_matrix.select(0, extra_channels_mapping[index]).clone())
            new_bias[index] = prev_bias[extra_channels_mapping[index]]

        if noise:
            noise = np.random.normal(scale=5e-5 * new_matrix.std(),
                                     size=list(new_matrix.size()))
            new_matrix += t.FloatTensor(noise).type_as(new_matrix)

        # assign new matrix and bias
        _assign_weights_and_bias_to_linear_(prev_layer, new_matrix.numpy(), new_bias.numpy())

    else:
        raise Exception("We can only handle input nn.Modules that are Linear or Conv2d at the moment.")


def _net2net_widen_input_channels_(next_layer, extra_channels, volume_slice_indxs, input_linear,
                                   new_hidden_units_per_new_channel,
                                   extra_channels_mapping, scaled):
    """
    Helper function for r2widerr. Containing all of the logic for widening the input channels of the 'next_layers'.
    
    :param next_layer: A layer after the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param volume_slice_indx: The indices to be able to slice the volume 
    :param extra_params_per_input: The number of extra parameters to add per layer that was input to the hidden volume.
    """
    # Check that we don't do linear -> conv, as haven't worked this out yet
    if input_linear and type(next_layer) is nn.Conv2d:
        raise Exception("We currently don't handle the nn.Linear -> nn.Conv2d case in r_2_wider_r.")

    if type(next_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        next_kernel = next_layer.weight.data.cpu()

        # Compute the new kernel for 'next_kernel' (extending each slice carefully)

        out_channels, in_channels, width, height = next_kernel.shape
        next_kernel_parts = []

        for i in range(1, len(volume_slice_indxs)):
            beg, end = volume_slice_indxs[i - 1], volume_slice_indxs[i]
            volume_extra_channels = extra_channels

            if scaled:
                volume_extra_channels = (end-beg) * (
                        extra_channels - 1)  # to triple number of channels, *add* 2x the current num

            original_kernel = next_kernel[:,beg:end]
            kernel_part = original_kernel.clone()
            kernel_part.resize_(out_channels, in_channels + volume_extra_channels, width, height)
            kernel_part = _net2net_extend_filter_input_channels(original_kernel, kernel_part,
                                                                in_channels, volume_extra_channels, extra_channels_mapping)

            next_kernel_parts.append(kernel_part)

        next_kernel = np.concatenate(next_kernel_parts, axis=1)

        # assign new conv (don't need to change bias)
        _assign_kernel_and_bias_to_conv_(next_layer, next_kernel)

    elif type(next_layer) is nn.Linear:

        # unpack linear params to numpy tensors
        next_matrix = next_layer.weight.data.cpu()
        # Compute the new matrix for 'next_matrix' (extending each slice carefully)
        n_out, n_in = next_matrix.shape

        next_matrix_parts = []
        for i in range(1, len(volume_slice_indxs)):
            beg = volume_slice_indxs[i - 1] * new_hidden_units_per_new_channel
            end = volume_slice_indxs[i] * new_hidden_units_per_new_channel
            volume_extra_channels = extra_channels

            if scaled:
                volume_extra_channels = (end-beg) * (extra_channels - 1)  # to triple number of channels, *add* 2x the current num

            original_matrix = next_matrix[:,beg:end]

            original_matrix.resize_(n_out, int(n_in/new_hidden_units_per_new_channel), new_hidden_units_per_new_channel)

            matrix_part = original_matrix.clone()
            matrix_part.resize_(n_out, int((n_in + volume_extra_channels)/new_hidden_units_per_new_channel), new_hidden_units_per_new_channel)

            matrix_part = _net2net_extend_filter_input_channels(original_matrix, matrix_part,
                                                                int(n_in / new_hidden_units_per_new_channel),
                                                                int(volume_extra_channels/new_hidden_units_per_new_channel),
                                                                extra_channels_mapping)

            matrix_part.resize_(n_out, (n_in + volume_extra_channels))

            next_matrix_parts.append(matrix_part)
        next_matrix = np.concatenate(next_matrix_parts, axis=1)

        # assign new linear params (don't need to change bias)
        _assign_weights_and_bias_to_linear_(next_layer, next_matrix)

    else:
        raise Exception("We can only handle output nn.Modules that are Linear or Conv2d at the moment.")


def _net2net_extend_filter_input_channels(original_kernel, kernel_part,
                                          in_channels, extra_channels, extra_channels_mapping):
    original_kernel = original_kernel.transpose(0, 1)
    kernel_part = kernel_part.transpose(0, 1)

    kernel_part.narrow(0, 0, in_channels).copy_(original_kernel)

    for index in range(in_channels, in_channels + extra_channels):
        kernel_part.select(0, index).copy_(original_kernel.select(0, extra_channels_mapping[index]).clone())

    original_channels_mapping = dict()
    for index in range(in_channels):
        original_channels_mapping[index] = [index]
    for key in extra_channels_mapping.keys():
        original_channels_mapping[extra_channels_mapping[key]].append(key)

    for _, values in original_channels_mapping.items():
        for value in values:
            kernel_part[value].div_(len(values))

    original_kernel.transpose_(0, 1)
    kernel_part.transpose_(0, 1)

    return kernel_part


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

    def __init__(self, input_shape):
        """
        Creates a graph object, with an initial node and no edges.
        """
        self.root_hvn = HVN(input_shape)
        self.nodes = [self.root_hvn]

    def add_hvn(self, hv_shape, input_modules=[], input_hvns=None, batch_norm=None):
        """
        Creates a new hvn node, and is just a wrapper around the HVN constructor.
        THe wrapper allows the graph object to keep track of the nodes that have been added.
        If input_hvn's aren't specified, assume that its the current set of output nodes from this graph.
        """
        if input_hvns is None:
            input_hvns = self.get_output_nodes()

        hvn = HVN(hv_shape, input_modules, input_hvns, batch_norm)
        self.nodes.append(hvn)
        return hvn

    def node_iterator(self):
        """
        Iterates through the nodes, returning tuples of ([prev_layer_modules], shape, batch_norm, [next_layer_modules]),
        ready to be fed into a volume widening function.

        Note that not all volumes are appropriate to be widened, only the one's that are both output from a layer and input
        to another layer.
        """
        for node in self.nodes:
            if len(node.child_edges) != 0 and len(node.parent_edges) != 0:
                yield (node._get_parent_modules(), node.hv_shape, node.batch_norm, node._get_child_modules())

    def get_output_nodes(self):
        """
        Gets a list of nodes that have no output edges
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


class HVN(object):
    """
    Class representing a node in the hidden volume graph.
    """

    def __init__(self, hv_shape, input_modules=[], input_hvns=[], batch_norm=None):
        """
        Initialize a node to be used in the hidden volume graph (HVG). It keeps track of a hidden volume shape and any
        associated batch norm layers.

        :param hv_shape: The shape of the volume being represented
        :param input_modules: the nn.Modules where input_modules[i] takes input with shape from input_hvns[i] to be
                concatenated for this hidden volume (node).
        :param input_hvns: A list of HVN objects that are parents/inputs to this HVN
        :param batch_norm: Any batch norm layer that's associated with this hidden volume, None if there is no batch norm
        """
        # Keep track of the state
        self.hv_shape = hv_shape
        self.child_edges = []
        self.parent_edges = []
        self.batch_norm = batch_norm

        # Make the edges between this node and it's parents/inputs
        self._link_nodes(input_modules, input_hvns)

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

    def _is_conv_volume(self):
        """
        Volumes output by conv layers are 3D, and linear are 1D.
        """
        return len(self.hv_shape) == 3


class HVE(object):
    """
    Class representing an edge in the hidden volume graph. It's basically a glorified pair object, with some reference to
    a nn.Module as well
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


def _hvg_from_lle(network):
    """
    Creates a HVG graph from the conv enum
    :param network: A nn.Module that implements the 'lle' (linear layer enum) interface
    """
    hvg = None
    prev_node = None
    prev_module = None
    for shape, batch_norm, module in network.lle():
        if not hvg:
            hvg = HVG(shape)
            prev_node = hvg.root_hvn
        else:
            prev_node = hvg.add_hvn(hv_shape=shape, input_modules=[prev_module],
                                    input_hvns=[prev_node], batch_norm=batch_norm)
        prev_module = module
    return hvg


def net2net_widen_network_(network, new_channels=0, new_hidden_nodes=0, scaled=True):
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
    :param scaled: If we want to extend channels by scaling (multiplying num channels) rather than adding
    :return: A reference to the widened network
    """
    #  Create the hidden volume graph
    if network.lle_or_hvg() == "lle":
        hvg = _hvg_from_lle(network)
    else:
        hvg = network.hvg()

    # Iterate through the hvg, widening appropriately in each place
    for prev_layers, shape, batch_norm, next_layers in hvg.node_iterator():
        linear_to_linear = type(prev_layers[0]) is nn.Linear and type(next_layers[0]) is nn.Linear
        channels_or_nodes_to_add = new_hidden_nodes if linear_to_linear else new_hidden_nodes
        if channels_or_nodes_to_add == 0:
            continue
        net_2_wider_net_(prev_layers, next_layers, shape, batch_norm, channels_or_nodes_to_add, scaled)

    # Return model for if someone want to use this in an assignment form etc
    return network


"""
Deepening networks.
"""


class Deepened_Network(nn.Module):
    """
    A heper class that encapsulates a "base" network and allows layers to be "inserted" into the middle of the
    base network.

    The base network must implement the following functions:

    Additionally, if we wish to maintain the ability to be widened, then the base network must implement the following:
    - base_network.conv_lle()
    - base_network.fc_lle()
    OR
    - base_network.conv_hvg()
    - base_network.fc_hvg(cur_hvg)
    """

    def __init__(self, base_network):
        """
        :param base_network: The base network that is to be "deepened"
        """
        super(Deepened_Network, self).__init__()

        if type(base_network) is Deepened_Network:
            raise Exception(
                "We shouldn't be recursively create Deepened_Network instances, it should instead be extended.")

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

        'module' should also be able to extend a hvg with a function 'conv_hvg(cur_hvg)'.

        Note: we take the liberty to assume that the shapings of layer are correct to just work.
        """
        if type(module) is nn.Linear:
            self.fc_extensions.append(module)
            self.add_module("fc_ext_{n}".format(n=len(self.fc_extensions)), module)
        else:
            self.conv_extensions.append(module)
            self.add_module("conv_ext_{n}".format(n=len(self.conv_extensions)), module)

    def lle_or_hvg(self):
        """
        To decide to use lle or hvg interface, just propogate what the base network was using.

        :return: String indicating if we're using "lle" or "hvg"
        """
        return self.base_network.lle_or_hvg()

    def hvg(self):
        """
        Implements the hvg interface, assuming that all components of the deepened network do too (base network and new
        modules).

        Assumes that the base network implements the following functions:
        self.base_network.input_shape()
        self.base_network.conv_hvg(cur_hvg)
        self.base_network.fc_hvg(cur_hvg)

        And assumes that all layers 'module' in self.conv_extensions implement:
        module.conv_hvg(cur_hvg)

        This function builds a hidden volume graph for the deepened network.
        """
        hvg = HVG(self.base_network.input_shape())
        hvg = self.base_network.conv_hvg(hvg)
        for conv_module in self.conv_extensions:
            hvg = conv_module.conv_hvg(hvg)
        hvg = self.base_network.fc_hvg(hvg)
        for fc_layer in self.fc_extensions:
            hvg.add_hvn((fc_layer.out_features,)[fc_layer])
        return hvg

    def lle(self):
        """
        Implements the lle interface, assuming that all components of the deepened network do too (base network and new
        modules).

        Assumes that the base network implements the following functions:
        self.base_network.conv_lle()
        self.base_network.fc_lle()

        It also assumes that each of the conv modules 'module' implements the following:
        module.lle()
        """
        itr = self.base_network.conv_lle()
        for conv_module in self.conv_extensions:
            itr = chain(itr, conv_module.conv_lle())
        itr = chain(itr, self.base_network.fc_lle())
        itr = chain(itr, self._fc_lle())
        return itr

    def _fc_lle(self):
        """
        Iterates through all of the fc additions in the deepening
        """
        for fc_layer in self.fc_extensions:
            yield ((fc_layer.in_features,), None, fc_layer)




def net2net_make_deeper_network_(network, layer):
    """
    Given a network 'network', create a deeper network adding in a new layer 'layer'.

    We assume the our network is build into a conv stack, which feeds into a fully connected stack.
    """
    if type(network) is not Deepened_Network:
        network = Deepened_Network(network)
    network.deepen(layer)
    return network




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
        self.bn = nn.BatchNorm2d(num_features=input_channels)

        self.update_conv()
        self.update_batch_norm()


    def update_conv(self, noise=True):

        conv_kernel = self.conv.weight.data.cpu()
        conv_bias = self.conv.bias.data.cpu()

        conv_kernel.zero_()
        conv_bias.zero_()

        center_height = (self.kernel_size[0] - 1) // 2
        center_width = (self.kernel_size[1] - 1) // 2

        for i in range(0, self.input_channel):
            conv_kernel.narrow(0, i, 1).narrow(1, i, 1).narrow(2, center_height, 1).narrow(3, center_width, 1).fill_(1)

        if noise:
            noise = np.random.normal(scale=5e-5,
                                     size=list(conv_kernel.size()))
            conv_kernel += t.FloatTensor(noise).type_as(conv_kernel)

        _assign_kernel_and_bias_to_conv_(self.conv, conv_kernel.numpy(), conv_bias.numpy())


    def update_batch_norm(self):

        new_scale = self.bn.weight.data.fill_(1)
        new_bias = self.bn.bias.data.fill_(0)
        new_running_mean = self.bn.running_mean.fill_(0)
        new_running_var = self.bn.running_var.fill_(1)

        _assign_to_batch_norm_(self.bn, new_scale.numpy(), new_bias.numpy(),
                               new_running_mean.numpy(), new_running_var.numpy())


    def forward(self, x):
        """
        Forward pass through this residual block

        :param x: the input
        :return: THe output of applying this residual block to the input
        """
        # Forward pass through residual part of the network

        x = self.conv(x)
        x = self.bn(x)
        #print ("After BN")
        #print (x2-x1)

        out = x

        return out

    def conv_lle(self):
        """
        Conv part of the 'lle' function (see below)
        """
        height, width = self.input_spatial_shape

        # 1st (not 0th) dimension is the number of in channels of a conv, so (in_channels, height, width) is input shape
        yield ((self.conv.weight.data.size(1), height, width), None, self.conv)


    def lle(self, input_shape=None):
        """
        Implement the lle (linear layer enum) to iterate through layers for widening.
        Input shape must either not be none here or not be none from before
        :param input_shape: Shape of the input volume, or, None if it wasn't already specified.
        :return: Iterable over the (in_shape, batch_norm, nn.Module)'s of the resblock
        """
        return self.conv_lle()

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

    def hvg(self):
        raise Exception("hvg() not implemented directly for net2net conv identity block.")








