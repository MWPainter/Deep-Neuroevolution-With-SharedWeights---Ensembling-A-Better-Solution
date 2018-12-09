import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from r2r.init_utils import *
from r2r.module_utils import *
from utils.pytorch_utils import cudafy

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
- network_instance.lle()
    This can be implemented *instead* of 'hvg'. A linear layer enumeration (lle), that returns an iterator that 
    sequentially gives (shape, batch_norm, nn.Module, Residual_Connection object) tuples, through the network. This can 
    only be used by networks that are completely linear and have one computation path through the network.  
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
    - network_instance.lle()
        Should just chain together the iterators from conv_lle() and fc_lle() below. (See examples - Mnist_Resnet).
    - network_instance.hvg()
        Should just chain together the building of conv_hvg() and fc_hvg() below. (See examples).
- all of R2DeeperR interface
- network_instance.lle_or_hvg():
    Should return 'lle' if the lle function is implemented, or 'hvg' if the hvg function is implemented instead.
- network_instance.conv_lle()
    If the network instance implements the lle function for R2WiderR, then it should implement conv_lle for the 
    convolutional part of it.
- network_instance.fc_lle()
    If the network instance implements the lle function for R2WiderR, then it should implement fc_lle for the 
    fully connected part of it.
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
- block.conv_lle()
    The nn.Module should implement a conv_lle function, for the lle interface, so that it can be used in future widening 
    operations, where the 'base_network' implemented the lle interface.
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




def _is_linear_volume(vol_shape):
    """
    Checks if a hidden volume is a "linear" volume
    """
    return len(vol_shape) == 1





def _is_conv_volume(vol_shape):
    """
    Checks if a hidden volume is a "convolutional" volume
    """
    return len(vol_shape) == 3





"""
Widening hidden volumes
"""




def r_2_wider_r_(prev_layers, next_layers, volume_shape, batch_norm=None, residual_connection=None, extra_channels=0,
                 init_type="He", function_preserving=True, multiplicative_widen=True):
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
    :param residual_connection: The Residual_Connection object associated with this volume. (I.e. if this volume x is
            used for a residual connection at some point later in the network y, with y = residual_connection(y,x)).
            This should be none if it's not used as part of a residual connection.
    :param input_shape: The shape of the input
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param function_preserving: If we wish for the widening to preserve the function I/O
    :param multiplicative_widen: If we want the number of extra channels to be multiplicative (i.e. 2 = double number
            of layers) or additive (i.e. 2 = add two layers)
    """
    # Handle inputting of single input/output layers
    if type(prev_layers) != list:
        prev_layers = [prev_layers]
    if type(next_layers) != list:
        next_layers = [next_layers]
        
    _r_2_wider_r_(prev_layers, next_layers, volume_shape, batch_norm, residual_connection, extra_channels, init_type,
                  function_preserving, multiplicative_widen)





def _r_2_wider_r_(prev_layers, next_layers, volume_shape, batch_norm, residual_connection, extra_channels, init_type,
                  function_preserving, multiplicative_widen):
    """  
    The full internal implementation of r_2_wider_r_. See description of r_2_wider_r_.
    More helper functions are used.
    """
    # For us to be able to perform a function preserving transforms, extra channels must be even and non-negative
    if extra_channels <= 0 or (function_preserving and extra_channels % 2 != 0):
        raise Exception("Invalid number of extra channels in widen.")
    
    # Check that the types of all the in/out layers are the same
    for i in range(1,len(prev_layers)):
        if type(prev_layers[0]) != type(prev_layers[i]):
            raise Exception("All input layers in R2WiderR need to be the same type, nn.Conv2D or nn.Linear")
    for i in range(1,len(next_layers)):
        if type(prev_layers[0]) != type(next_layers[i]):
            raise Exception("All output layers in R2WiderR need to be the same type, nn.Conv2D or nn.Linear")
            
    # Check that the volume shape is either linear or convolutional
    if len(volume_shape) not in [1,3]:
        raise Exception("Volume shape must be 1D or 3D for R2WiderR to work")
            
    # Get if we have a linear input or not
    input_is_linear = type(prev_layers[0]) is nn.Linear
            
    # Work out the number of hiden units per new channel 
    # (for fc pretend 1x1 spatial resolution, so 1 per channel) (for conv this is width*height)
    # For conv -> linear layers this can be a little complex. But we always know the number of channels from the prev kernels
    channels_in_volume = np.sum([layer.weight.size(0) for layer in prev_layers])
    total_hidden_units = np.prod(volume_shape)
    new_hidden_units_per_new_channel = total_hidden_units // channels_in_volume
    
    # Sanity check
    if input_is_linear and new_hidden_units_per_new_channel != 1:
        raise Exception("Number of 'new hidden_units per new channel' must be 1 for linear. Something went wrong :(.")
    
    # Compute the slicing of the volume from the input (to widen outputs appropraitely)
    volume_slices_indices = [0]
    for prev_layer in prev_layers:
        new_slice_indx = volume_slices_indices[-1] + prev_layer.weight.size(0)
        volume_slices_indices.append(new_slice_indx)
    
    # Sanity check that our slices are covering the entire volume that is being widened
    # There is some complexity when conv volumes are flattened as input to a linear layer
    # We effectively check that the input is consistent with the volume shape here
    if ((_is_conv_volume(volume_shape) and volume_slices_indices[-1] != volume_shape[0]) or
        (_is_linear_volume(volume_shape) and volume_slices_indices[-1] * new_hidden_units_per_new_channel != volume_shape[0])):
        raise Exception("The shape output from the input layers is inconsistent with the hidden volume provided in R2WiderR.")
    
    # Iterate through all of the prev layers, and widen them appropraitely
    for prev_layer in prev_layers:
        _widen_output_channels_(prev_layer, extra_channels, init_type, multiplicative_widen)
    
    # Widen batch norm appropriately 
    if batch_norm:
        _extend_bn_(batch_norm, extra_channels, volume_slices_indices, multiplicative_widen)

    # Widen the residual connection appropriately
    if residual_connection:
        residual_connection._widen_(volume_slices_indices, extra_channels, multiplicative_widen)
    
    # Iterate through all of the next layers, and widen them appropriately. (Needs the slicing information to deal with concat)
    for next_layer in next_layers:
        _widen_input_channels_(next_layer, extra_channels, init_type, volume_slices_indices, input_is_linear,
                               new_hidden_units_per_new_channel, multiplicative_widen, function_preserving)
        
        
        


def _extend_bn_(bn, new_channels_per_slice, volume_slices_indices, multiplicative_widen):
    """
    Extend batch norm with 'new_channels' many extra units. Initialize values to zeros and ones appropriately.
    Really this is just a helper function for R2WiderR
    
    :param bn: The batch norm layer to be extended
    :param new_channels_per_slice: The number of new channels to add, per slice
    :param volume_slices_indices: The indices to be able to slice the volume (so that we can add new
    :param multiplicative_widen: If true, then we should interpret "extra channels" as a multiplicative factor for the
            number of outputs (rather than additive)
    """
    new_scale_slices = []
    new_shift_slices = []
    new_running_mean_slices = []
    new_running_var_slices = []
    
    old_scale = bn.weight.data.cpu().numpy()
    old_shift = bn.bias.data.cpu().numpy()
    old_running_mean = bn.running_mean.data.cpu().numpy()
    old_running_var = bn.running_var.data.cpu().numpy()
    
    for i in range(1,len(volume_slices_indices)):
        beg = volume_slices_indices[i-1]
        end = volume_slices_indices[i]
        
        # to triple number of channels, *add* 2x the current num
        new_channels = new_channels_per_slice if not multiplicative_widen else (new_channels_per_slice - 1) * (end-beg)
        
        new_scale_slices.append(_one_pad_1d(old_scale[beg:end], new_channels))
        new_shift_slices.append(_zero_pad_1d(old_shift[beg:end], new_channels))
        new_running_mean_slices.append(_zero_pad_1d(old_running_mean[beg:end], new_channels))
        new_running_var_slices.append(_zero_pad_1d(old_running_var[beg:end], new_channels))
        
    new_scale = np.concatenate(new_scale_slices)
    new_shift = np.concatenate(new_shift_slices)
    new_running_mean = np.concatenate(new_running_mean_slices)
    new_running_var = np.concatenate(new_running_var_slices)
        
    _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)
    
    
    
    
    
def _widen_output_channels_(prev_layer, extra_channels, init_type, multiplicative_widen):
    """
    Helper function for r2widerr. Containing all of the logic for widening the output channels of the 'prev_layers'.
    
    :param prev_layer: A layer before the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param multiplicative_widen: If true, then we should interpret "extra channels" as a multiplicative factor for the
            number of outputs, rather than additive
    """
    if type(prev_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        prev_kernel = prev_layer.weight.data.cpu().numpy()
        prev_bias = prev_layer.bias.data.cpu().numpy()
        
        # new conv kernel
        old_out_channels, in_channels, width, height = prev_kernel.shape
        if multiplicative_widen:
            extra_channels = old_out_channels * (extra_channels - 1) # to triple number of channels, *add* 2x the current num
        kernel_extra_shape = (extra_channels, in_channels, width, height)
        prev_kernel = _extend_filter_with_repeated_out_channels(kernel_extra_shape, prev_kernel, init_type)

        # zero pad the bias
        prev_bias = _zero_pad_1d(prev_bias, extra_channels)
        
        # assign new conv and bias
        _assign_kernel_and_bias_to_conv_(prev_layer, prev_kernel, prev_bias)
        
    elif type(prev_layer) is nn.Linear:
        # unpack linear params to numpy tensors
        prev_matrix = prev_layer.weight.data.cpu().numpy()
        prev_bias = prev_layer.bias.data.cpu().numpy()
        
        # new linear matrix
        old_n_out, n_in = prev_matrix.shape
        if multiplicative_widen:
            extra_channels = old_n_out * (extra_channels - 1) # to triple number of channels, *add* 2x the current num
        matrix_extra_shape = (extra_channels, n_in)
        prev_matrix = _extend_matrix_with_repeated_out_weights(matrix_extra_shape, prev_matrix, init_type)

        # zero pad the bias
        prev_bias = _zero_pad_1d(prev_bias, extra_channels)
        
        # assign new matrix and bias
        _assign_weights_and_bias_to_linear_(prev_layer, prev_matrix, prev_bias)
        
    else:
        raise Exception("We can only handle input nn.Modules that are Linear or Conv2d at the moment.")
                            
                            
                            
                            
                            
def _widen_input_channels_(next_layer, extra_channels, init_type, volume_slices_indices, input_is_linear,
                           new_hidden_units_per_new_channel, multiplicative_widen, function_preserving):
    """
    Helper function for r2widerr. Containing all of the logic for widening the input channels of the 'next_layers'.
    
    :param next_layer: A layer after the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param volume_slices_indices: The indices to be able to slice the volume
    :param input_is_linear: If input is from a linear layer previously
    :param new_hidden_units_per_new_channel: The number of new hidden units in the volume per channel output from 
            'prev_layers'. Conssider a linear 'prev_layer' to be a 1x1 conv, on a volume with 1x1 spatial dimensions.
    :param multiplicative_widen: If true, then we should interpret "extra channels" as a multiplicative factor for the
            number of outputs, rather than additive
    :param function_preserving: If we want the widening to be done in a function preserving fashion
    """
    # Check that we don't do linear -> conv, as haven't worked this out yet
    if input_is_linear and type(next_layer) is nn.Conv2d:
        raise Exception("We currently don't handle the nn.Linear -> nn.Conv2d case in r_2_wider_r.")
    
    if type(next_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        next_kernel = next_layer.weight.data.cpu().numpy()
        next_bias = next_layer.bias.data.cpu().numpy()
        
        # Compute the new kernel for 'next_kernel' (extending each slice carefully)
        alpha = -1.0 if function_preserving else 1.0
        out_channels, old_in_channels, width, height = next_kernel.shape 
        next_kernel_parts = []
        for i in range(1, len(volume_slices_indices)):
            beg, end = volume_slices_indices[i-1], volume_slices_indices[i]
            if multiplicative_widen:
                slice_extra_channels = (end-beg) * (extra_channels - 1) # to triple num channels, *add* 2x the current num
            kernel_extra_shape = (out_channels, slice_extra_channels, width, height)
            kernel_part = _extend_filter_with_repeated_in_channels(kernel_extra_shape, next_kernel[:,beg:end], 
                                                                   init_type, alpha)
            next_kernel_parts.append(kernel_part)
        next_kernel = np.concatenate(next_kernel_parts, axis=1)
        
        # assign new conv (don't need to change bias)
        _assign_kernel_and_bias_to_conv_(next_layer, next_kernel)
        
    elif type(next_layer) is nn.Linear:            
        # unpack linear params to numpy tensors
        next_matrix = next_layer.weight.data.cpu().numpy()
        next_bias = next_layer.bias.data.cpu().numpy()
        
        # Compute the new matrix for 'next_matrix' (extending each slice carefully)
        alpha = -1.0 if function_preserving else 1.0
        n_out, old_n_in = next_matrix.shape 
        next_matrix_parts = []
        for i in range(1, len(volume_slices_indices)):
            beg = volume_slices_indices[i-1] * new_hidden_units_per_new_channel
            end = volume_slices_indices[i] * new_hidden_units_per_new_channel
            if multiplicative_widen:
                extra_params_per_input_layer = (end-beg) * (extra_channels - 1) # triple num outputs = *add* 2x the current num
            matrix_extra_shape = (n_out, extra_params_per_input_layer)
            matrix_part = _extend_matrix_with_repeated_in_weights(matrix_extra_shape, next_matrix[:,beg:end], init_type, alpha)
            next_matrix_parts.append(matrix_part)
        next_matrix = np.concatenate(next_matrix_parts, axis=1)        
        
        # assign new linear params (don't need to change bias)
        _assign_weights_and_bias_to_linear_(next_layer, next_matrix)
        
    else:
        raise Exception("We can only handle output nn.Modules that are Linear or Conv2d at the moment.")
        


        
        
        






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
        if input_hvns is None:
            input_hvns = self.get_output_nodes()
        
        hvn = HVN(hv_shape, input_modules, input_hvns, batch_norm, residual_connection)
        self.nodes.append(hvn)
        return hvn
    
    
    
    def node_iterator(self):
        """
        Iterates through the nodes, returning tuples of ([prev_layer_modules], shape, batch_norm,
        residual connection object, [next_layer_modules]), ready to be fed into a volume widening function.

        Note that not all volumes are appropriate to be widened, only the one's that are both output from a layer and input
        to another layer.

        :yields: ([list of parent nn.Modules (edges)], hidden volume shape, batch norm, residual connection object,
                    [list of child nn.Modules (edges)]) tuples.
        """
        for node in self.nodes:
            if len(node.child_edges) != 0 and len(node.parent_edges) != 0:
                yield (node._get_parent_modules(), node.hv_shape, node.batch_norm, node.residual_connection, node._get_child_modules())
            
            
            
    def get_output_nodes(self):
        """
        Gets a list of nodes that have no output edges

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
    
    
    
class HVN(object):
    """
    Class representing a node in the hidden volume graph.
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
        :param batch_norm: Any batch norm layer that's associated with this hidden volume, None if there is no batch norm
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
        
        
        
        
        
def _hvg_from_lle(network):
    """
    Creates a HVG graph from the conv enum 
    :param network: A nn.Module that implements the 'lle' (linear layer enum) interface
    :returns: A HVG object created using the lle() interface.
    """
    hvg = None
    prev_node = None
    prev_module = None
    for shape, batch_norm, module, residual_connection in network.lle():
        if not hvg:
            hvg = HVG(shape, residual_connection=residual_connection)
            prev_node = hvg.root_hvn
        else:
            prev_node = hvg.add_hvn(hv_shape=shape, input_modules=[prev_module], input_hvns=[prev_node],
                                    batch_norm=batch_norm, residual_connection=residual_connection)
        prev_module = module
    return hvg
        
        
        

        
def widen_network_(network, new_channels=0, new_hidden_nodes=0, init_type='He', function_preserving=True,
                   multiplicative_widen=True):
    """
    We implement a loop that loops through all the layers of a network, according to what we will call the 
    enum_layers interface. The interface should return, for every hidden volume, the layer before it and 
    any batch norm layer associated with it, along with the shape for the current hidden volume.
    
    For simplicity we add 'new_channels' many new channels to hidden volumes, and 'new_units' many additional 
    hidden units. By setting one of them to zero we can easily just widen the convs or fc layers of a network independently.
    
    :param network: The nn.Module to be widened, which must implement the R2WiderR interface.
    :param new_channels: The number of new channels/hidden units to add in conv layers
    :param new_hidden_nodes: The number of new hidden nodes to add in fully connected layers
    :param init_type: The initialization type to use for new variables
    :param function_preserving: If we want the widening to be function preserving
    :param multiplicative_widen: If we want to extend channels by scaling (multiplying num channels) rather than adding
    :return: A reference to the widened network
    """
    # Create the hidden volume graph
    if network.lle_or_hvg() == "lle":
        hvg = _hvg_from_lle(network)
    else:
        hvg = network.hvg()
        
    # Iterate through the hvg, widening appropriately in each place
    for prev_layers, shape, batch_norm, residual_connection, next_layers in  hvg.node_iterator():
        feeding_to_linear = (type(prev_layers[0]) is nn.Linear) or (type(next_layers[0]) is nn.Linear)
        channels_or_nodes_to_add = new_hidden_nodes if feeding_to_linear else new_channels
        if channels_or_nodes_to_add == 0:
            continue
        r_2_wider_r_(prev_layers, next_layers, shape, batch_norm, residual_connection, channels_or_nodes_to_add,
                     init_type, function_preserving, multiplicative_widen)
        
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
    
    
    
    def lle(self):
        """
        Implements the lle interface, assuming that all components of the deepened network do too (base network and new 
        modules).
        
        Assumes that the base network implements the following functions:
        self.base_network.conv_lle()
        self.base_network.fc_lle()
        
        It also assumes that each of the conv modules 'module' implements the following:
        module.lle()

        :yields: Tuples of (shape, batchnorm, nn.Module, Residual_Connection object) from linearly scanning over the
                network.
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

        :yields: The lle for the fully connected portion of the network.
        """
        for fc_layer in self.fc_extensions:
            yield ((fc_layer.in_features,), None, fc_layer)

        
        
        
        
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


