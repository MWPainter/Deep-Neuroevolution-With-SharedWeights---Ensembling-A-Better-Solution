import torch as t
import torch.nn as nn
import torch.nn.functional as F

from init_utils.py import *
from module_utils.py import *





"""
This file contains all of the logic related to performing widening and deepening transforms and initializations.

For a nn.Module to be widened it needs to implement the following interface for *linear layer enumeration*:
network_instance.lle()

OR a more complex nn.Module involving more than a linear sequence of nn.Modules (i.e. iff it uses concatenation) must 
implement the following interface for *hidden volume graph* creation:
network_instance.hvg()

For a nn.Module to be deepened, it needs to implement the following interface:
network_instance.conv_forward(x) - the conv portion of the network
network_instance.fc_forward(x)   - the fc portion of the network
network_instance.out_forward(x)  - the output portion of the network (anything that happens after the fc networks)

If we wish for a nn.Module to ever be widened after a deepen transform, then, we must implement the following additional 
functions, which break down the computation of :
network_instance.conv_lle()
network_instance.fc_lle()
OR
network_instance.conv_hvg()
network_instance.fc_hvg()


We use the following python and PyTorch conventions for function naming very explicitly:
_function_name = a private/helper function, not to be exported;
function_name_ = a function that has side effects (will change the objects that have been passed into them).

Note, that this means that we will sometimes be declaring function names like '_function_name_', when both cases are true.
"""





# Only export things that actually widen/deepen volumes, and not helper functions
all = [r_2_wider_r_,
       HVG, 
       HVN, 
       HVE, 
       widen_network_,
       Deepened_Network, 
       make_deeper_network] # make_deeper_network = r2deeperr if add identity initialized module





"""
Widening hidden volumes
"""




def r_2_wider_r_(prev_layers, next_layers, volume_shape, extra_channels=0, init_type="He", function_preserving=True):
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
        
    _r_2_wider_r_(prev_layers, next_layers, volume_shape, batch_norm, extra_channels, init_type, function_preserving)





def _r_2_wider_r_(prev_layers, next_layers, volume_shape, batch_norm, extra_channels, init_type, function_preserving):
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
            
    # Work out the number of params per new channel (for fc this is 1)
    new_params_per_new_channel = 1
    if len(volume_shape) == 3:
        _, height, width = volume_shape
        new_params_per_new_channel = height * width
    
    # Compute the slicing of the volume from the input (to widen outputs appropraitely)
    volume_slices_indxs = [0]
    for prev_layer in prev_layers:
        new_slice_indx = volume_slices_indxs[-1] + prev_layer.weight.size(0)
        volume_slices_indxs.append(new_slice_indx)
    
    # Sanity check that our slices are covering the entire volume that is being widened
    if volume_slices_indxs[-1] != volume_shape[0]:
        raise Exception("Number of channels output from prev_layers is inconsistent with the number of channels in volume " +
                        "shape. They should be consistent.")
    
    # Iterate through all of the prev layers, and widen them appropraitely
    for prev_layer in prev_layers:
        _widen_output_channels_(prev_layer, extra_channels, init_type)
    
    # Widen batch norm appropriately 
    if batch_norm:
        _extend_bn_(batch_norm, extra_channels)
    
    # Iterate through all of the next layers, and widen them appropriately. (Needs the slicing information to deal with concat)
    for next_layer in next_layers:
        _widen_input_channels_(next_layer, extra_channels, init_type, volume_slice_indxs)

        
        
        

def _extend_bn_(bn, new_channels):
    """
    Extend batch norm with 'new_channels' many extra units. Initialize values to zeros and ones appropriately.
    Really this is just a helper function for R2WiderR
    
    :param bn: The batch norm layer to be extended
    :param new_channels: The number of new channels to add
    """
    new_scale = _one_pad_1d(bn.weight.data.cpu().numpy(), new_channels)
    new_shift = _zero_pad_1d(bn.bias.data.cpu().numpy(), new_channels)
    new_running_mean = _zero_pad_1d(bn.running_mean.data.cpu().numpy(), new_channels)
    new_running_var = _zero_pad_1d(bn.running_var.data.cpu().numpy(), new_channels)
        
    _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)
    
    
    
    
    
def _widen_output_channels_(prev_layer, extra_channels, init_type):
    """
    Helper function for r2widerr. Containing all of the logic for widening the output channels of the 'prev_layers'.
    
    :param prev_layer: A layer before the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    """
    if type(prev_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        prev_kernel = prev_layer.weight.data.cpu().numpy()
        prev_bias = prev_layer.bias.data.cpu().numpy()
        
        # new conv kernel
        _, in_channels, width, height = prev_kernel.shape
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
        _, n_in = prev_matrix.shape
        matrix_extra_shape = (extra_channels, n_in)
        prev_matrix = _extend_matrix_with_repeated_out_weights(matrix_extra_shape, prev_matrix, init_type)

        # zero pad the bias
        prev_bias = _zero_pad_1d(prev_bias, extra_channels)
        
        # assign new matrix and bias
        _assign_weights_and_bias_to_linear_(prev_layer, prev_matrix, prev_bias)
        
    else:
        raise Exception("We can only handle input nn.Modules that are Linear or Conv2d at the moment.")
                            
                            
                            
                            
                            
def _widen_input_channels_(next_layer, extra_channels, init_type, volume_slice_indxs, input_linear, extra_params_per_input):
    """
    Helper function for r2widerr. Containing all of the logic for widening the input channels of the 'next_layers'.
    
    :param next_layer: A layer after the hidden volume/units being widened
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param volume_slice_indx: The indices to be able to slice the volume 
    :param extra_params_per_input: The number of extra parameters to add per layer that was input to the hidden volume.
    """
    # Check that we don't do linear -> conv, as haven't worked this out yet
    if input_linear and type(next_layer) is nn.Conv2d:
        raise Exception("We currently don't handle the nn.Linear -> nn.Conv2d case in r_2_wider_r.")
    
    if type(next_layer) is nn.Conv2d:
        # unpack conv2d params to numpy tensors
        next_kernel = next_layer.weight.data.cpu().numpy()
        next_bias = next_layer.bias.data.cpu().numpy()
        
        # Compute the new kernel for 'next_kernel' (extending each slice carefully)
        alpha = -1.0 if function_preserving else 1.0
        out_channels, _, width, height = next_kernel.shape 
        kernel_extra_shape = (out_channels, extra_channels, width, height)
        next_kernel_parts = []
        for i in range(1, len(volume_slice_indxs)):
            beg, end = volume_slice_indxs[i-1], volume_slice_indxs[i]
            kernel_part = _extend_filter_with_repeated_in_channels(kernel_extra_shape, next_kernel[beg:end], init_type, alpha)
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
        n_out, _ = next_matrix.shape 
        matrix_extra_shape = (n_out, extra_params_per_input)
        next_matrix_parts = []
        for i in range(1, len(volume_slice_indxs)):
            beg, end = volume_slice_indxs[i-1], volume_slice_indxs[i]
            matrix_part = _extend_matrix_with_repeated_in_weights(matrix_extra_shape, next_matrix[beg:end], init_type, alpha)
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
    def __init__(self, input_shape):
        """
        Creates a graph object, with an initial node and no edges.
        """
        self.root_hvn = HVN(input_shape)
        self.nodes = [self.root_hvn]
    
    
    
    def add_hvn(self, hv_shape, input_hvns=[], input_modules=[], batch_norm=None):
        """
        Creates a new hvn node, and is just a wrapper around the HVN constructor. 
        THe wrapper allows the graph object to keep track of the nodes that have been added.
        """
        hvn = HVN(hv_shape, input_hvns, input_modules, batch_norm)
        self.nodes.append(hvn)
        return hvn
    
    
    
    def node_iterator(self):
        """
        Iterates through the nodes, returning tuples of ([prev_layer_modules], shape, batch_norm, [next_layer_modules]),
        ready to be fed into a volume widening function.
        """
        for node in self.nodes:
            yield (node._get_parent_modules(), node.hv_shape, node.batch_norm, node._get_child_modules())
    
    
    
    
class HVN(object):
    """
    Class representing a node in the hidden volume graph.
    """
    def __init__(self, hv_shape, input_hvns=[], input_modules=[], batch_norm=None):
        """
        Initialize a node to be used in the hidden volume graph (HVG). It keeps track of a hidden volume shape and any 
        associated batch norm layers.
        
        :param hv_shape: The 
        :param input_hvns: A list of HVN objects that are parents/inputs to this HVN
        :param input_modules: the nn.Modules where input_modules[i] takes input with shape from input_hvns[i] to be 
                concatenated for this hidden volume (node).
        :param batch_norm: Any batch norm layer that's associated with this hidden volume, None if there is no batch norm
        """
        # Keep track of the state
        self.hv_shape = hv_shape
        self.child_edges = []
        self.parent_edges = []
        self.batch_norm = batch_norm
        
        # Make the edges between this node and it's parents/inputs
        self._link_nodes(input_hvns, input_modules)
        
        
    
    def _link_nodes(self, input_hvns, input_modules):
        """
        Links a list of hvn's to this hvn, with edges that correspond to PyTorch nn.Modules
        
        :param input_hvns: A list of HVN objects that are parents/inputs to this HVN
        :param input_modules: the nn.Modules where input_modules[i] takes input with shape from input_hvns[i] to be 
                concatenated for this hidden volume (node).
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
        return [edge.module for edge in self.parent_edges]
    
    
    
    def _get_child_modules(self):
        """
        Gets a list of nn.Modules from the child edges (in the same order)
        """
        return [edge.module for edge in self.child_edges]
    
    
    
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
        
        
        
        
        
def _hvg_from_enum(network):
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
            prev_node = hvg.add_hvn(shape, [prev_node], [prev_module], batch_norm)
        prev_module = module
    return hvg
        
        
        

        
def widen_network_(network, new_channels=0, new_hidden_nodes=0, init_type='He', function_preserving=True):
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
    :return: A reference to the widened network
    """
    # Create the hidden volume graph
    if network.lle:
        hvg = _hvg_from_enum(network)
    else:
        hvg = network.hvg()
        
    # Iterate through the hvg, widening appropriately in each place
    for prev_layers, shape, batch_norm, next_layers in  hvg.node_iterator():
        channels_or_nodes_to_add = new_channels if len(shape) == 3 else new_hidden_nodes
        if channels_or_nodes_to_add == 0:
            continue
        r_2_wider_r_(prev_layers, next_layers, shape, channels_or_nodes_to_add, init_type, function_preserving)




        
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
    - base_network.fc_hvg()
    """
    def __init__(self, base_network):
        """
        :param base_network: The base network that is to be "deepened"
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
        for conv_layer in self.conv_extensions: 
            x = conv_layer(x)
        x = self.base_network.fc_forward(x)
        for fc_layer in self.fc_extensions:
            x = fc_layer(x)
        return self.base_network.out_forward(x)
        
        
    def deepen(self, layer):
        """
        Extends the network with nn.Module 'layer'. We assume that the layer is to be applied between the convolutional 
        stack and the fc stack, unless 'layer' is of type nn.Linear.
        
        To register the parameters for autograd computation, we need to use "add_module", which is similar to 
        "register_parameter".
        
        Note: we take the liberty to assume that the shapings of layer are correct to just work.
        """
        if type(layer) is nn.Linear:
            self.fc_extensions.append(layer)
            self.add_module("fc_ext_{n}".format(n=len(self.fc_extensions)), layer)
        else:
            self.conv_extensions.append(layer)
            self.add_module("conv_ext_{n}".format(n=len(self.conv_extensions)), layer)

        
        
        
        
def make_deeper_network(network, layer):
    """
    Given a network 'network', create a deeper network adding in a new layer 'layer'. 
    
    We assume the our network is build into a conv stack, which feeds into a fully connected stack. 
    """
    if type(network) is not Deepened_Network:
        network = Deepened_Network(network)
    network.extend(layer)
    return network

