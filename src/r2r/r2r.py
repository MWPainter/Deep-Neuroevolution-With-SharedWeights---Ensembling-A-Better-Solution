import torch as t
import torch.nn as nn
import torch.nn.functional as F

from init_utils.py import *
from module_utils.py import *




"""
TODO:
1. Move the commented out functions below into the correct place (taking a graph as input)
2. Making graphs from the enum functions
3. Updating the implementation of _r_2_wider_r_
4. Update all = [...]
"""





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
all = [HVG, HVN, HVE, Deepened_Network, make_deeper_network]





"""
Widening hidden volumes
"""





def r_2_wider_r_no_concat_(prev_layer, next_layer, volume_shape, extra_channels=0, init_type='He', function_preserving=True):
    """
    Helper to still provide a consistent interface as before
    """
    return r_2_wider_r_([prev_layer], [next_layer], volume_shape, extra_channels, init_type, function_preserving)





# TODO (make it work for the 3 changed inputs)
def r_2_wider_r_(prev_layers, next_layers, volume_shape, extra_channels=0, init_type='He', function_preserving=True):
    """
    This is an almost complete implementation of R2WiderR. All that will remain is dealing with lists of layers
    for input and output, for use with Inception networks.
    
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
    
    :param prev_layer: A list of layers before the hidden volume/units being widened (their outputs are concatenated together)
    :param next_layer: A list of layers layer after the hidden volume/units being widened (inputs are the concatenated out)
    :param input_shape: The shape of the input
    :param extra_channels: The number of new conv channels/hidden units to add
    :param init_type: The type of initialization to use ('He' or 'Xavier')
    :param function_preserving: If we wish for the widening to preserve the function I/O
    :return: 
    """
    # For us to be able to perform a function preserving transforms, extra channels must be even and non-negative
    if extra_channels <= 0 or (function_preserving and extra_channels % 2 != 0):
        raise Exception("Invalid number of extra channels in widen.")
        
    new_params_shape = None
    
    # Compute the widened output of prev layer, and assign the new variables to the layer
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
        
        # Compute the shape of the new hidden volume added
        _, img_width, img_height = input_shape
        pad_width, pad_height = prev_layer.padding
        stride_width, stride_height = prev_layer.stride
        new_img_width = (img_width - width + 2*pad_width) / stride_width
        new_img_height = (img_height - height + 2*pad_height) / stride_height
        new_params_shape = (extra_channels, new_img_width, new_img_height)
        
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
        
        # Compute the shape of the new hidden units added
        new_params_shape = (extra_channels)
        
    else:
        raise Exception("We can only handle input nn.Modules that are Linear or Conv2d at the moment.")
        
       
    # Compute the widened input of next layer, and assign the new variables to the layer 
    if type(next_layer) is nn.Conv2d:
        # Check that we only do conv to conv, and not linear to conv
        if len(shape) != 4:
            raise Exception("We currently don't handle the nn.Linear -> nn.Conv2d case.")
            
        # unpack conv2d params to numpy tensors
        next_kernel = next_layer.weight.data.cpu().numpy()
        next_bias = next_layer.bias.data.cpu().numpy()
        
        # Compute the new kernel for 'next_kernel'
        alpha = -1.0 if function_preserving else 1.0
        out_channels, _, width, height = next_kernel.shape 
        kernel_extra_shape = (out_channels, extra_channels, width, height)
        next_kernel = _extend_filter_with_repeated_in_channels(kernel_extra_shape, next_kernel, init_type, alpha)
        
        # assign new conv (don't need to change bias)
        _assign_kernel_and_bias_to_conv_(next_layer, next_kernel)
        
    elif type(next_layer) is nn.Linear:
        # The number of extra inputs is the total flattened size
        extra_params = np.prod(new_params_shape)
            
        # unpack linear params to numpy tensors
        next_matrix = next_layer.weight.data.cpu().numpy()
        next_bias = next_layer.bias.data.cpu().numpy()
        
        # Compute the new matrix for 'next_matrix'
        alpha = -1.0 if function_preserving else 1.0
        n_out, _ = next_matrix.shape 
        matrix_extra_shape = (n_out, extra_params)
        next_matrix = _extend_matrix_with_repeated_in_weights(matrix_extra_shape, next_matrix, init_type, alpha)
        
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






"""
TODO: some helper functions to widen entire networks
"""
# def _extend_bn_(bn, new_channels):
#     """
#     Extend batch norm with 'new_channels' many extra units. Initialize values to zeros and ones appropriately.
#     Really this is just a helper function for R2WiderR
    
#     :param bn: The batch norm layer to be extended
#     :param new_channels: The number of new channels to add
#     """
#     new_scale = _one_pad_1d(bn.weight.data.cpu().numpy(), new_channels)
#     new_shift = _zero_pad_1d(bn.bias.data.cpu().numpy(), new_channels)
#     new_running_mean = _zero_pad_1d(bn.running_mean.data.cpu().numpy(), new_channels)
#     new_running_var = _zero_pad_1d(bn.running_var.data.cpu().numpy(), new_channels)
        
#     _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)
        


        
        
# def _widen_network_all_layers_(network, extra_channels, init_type='He', function_preserving=True):
#     """
#     We implement a loop that loops through all the layers of a network, according to what we will call the 
#     enum_layers interface. The interface should return, for every hidden volume, the layer before it and 
#     any batch norm layer associated with it, along with the shape for the current hidden volume.
    
#     For simplicity we add 'exp_channels' many new channels to hidden volumes, and 'exp_channels' many additional 
#     hidden units. 
    
#     To be explicit abour how 'enum_layers' should be implemented:
#     for each hidden_volume (*including*) the input volume:
#         return the shape of it
#         return the batch norm for it (none if there is not a batch norm)
#         return the layer that it feeds into
    
#     :param network: The nn.Module to be widened
#     :param extra_channels: The number of new channels/hidden units to add
#     :param init_type: The initialization type to use for new variables
#     :param function_preserving: If we want the widening to be function preserving
#     :return: A reference to the widened network
#     """
#     prev_layer, prev_bn, prev_shape, next_layer, next_bn, next_shape = None, None, None, None, None, None
    
#     # Iterate through all hidden voluems of the network
#     for shape, bn, layer in network.enum_layers():
#         prev_shape = next_shape    # shape input to layer before hidden volume being extended
#         prev_bn = next_bn          # bn for the layer before (can be ignored)
#         prev_layer = next_layer    # the layer before the hidden volume being extended
        
#         next_shape = shape         # the shape of the volume being extended
#         next_bn = bn               # the bn for the current layer
#         next_layer = layer         # the layer the hidden volume is fed into
        
#         # Skip on the first one
#         if prev_layer is None:
#             continue
            
#         # Apply R2WiderR at this hidden volume
#         _r_2_wider_r_(prev_layer, next_layer, prev_shape, extra_channels, init_type, function_preserving)
        
#         # Extend the batch norms at this hidden volume too
#         if next_bn is not None:
#             _extend_bn_(next_bn, extra_channels)






# def _widen_network_convs_(network, exp_channels, init_type='He', function_preserving=True, uniform=False):
#     """
#     A common implementation of 'uniform_widen_network_conv' and 'scaled_widen_network_conv'.
#     Iterate through all of the convolutional and batch norm layers, extending the number of channels in each
#     volume by 'new_channels'.
    
#     Note that the one volume it doesn't change in the convolutional stack is the output, because it 
#     needs a consistent shape (for now) to be input into the fully connected portion of the network.
    
#     :param network: A nn.Module that implements the methods 'conv_enum' and 'batch_norm_enum' to widen
#     :param exp_channels: The number of new channels to add if 'uniform', otherwise, it's a multiplicative constant
#             to scale the number of channels by.
#     :param init_type: Either 'He' or 'Xavier'. Specifies the init type to use for new params.
#     :param function_preserving: If we want to initialize new parameters for a function preserving transform.
#     :param uniform: If we should treat 'exp_channels' as the number of new channels to add at every volume, 
#             or, if we should treat it as a multiplicative constant of how many channels to add.
#     """
#     prev_conv, next_conv = None, None
#     for conv in network.conv_enum():
#         prev_conv = next_conv
#         next_conv = conv
        
#         # Skip on the first one
#         if prev_conv is None:
#             continue
            
#         # Get numpy params
#         prev_kernel = prev_conv.weight.data.cpu().numpy()
#         prev_bias = prev_conv.bias.data.cpu().numpy()
#         next_kernel = next_conv.weight.data.cpu().numpy()
        
#         # Compute widening (note if multiplying channels by m, we only want (m-1)*old_channels additional channels)
#         new_channels = exp_channels if uniform else prev_kernel.shape[0] * (exp_channels - 1)
#         new_params = _widen_hidden_volume(prev_kernel, prev_bias, next_kernel, extra_channels=new_channels,
#                                           init_type=init_type, function_preserving=function_preserving)
#         prev_kernel, prev_bias, next_kernel = new_params
        
#         # Make assignment
#         _assign_kernel_and_bias_to_conv_(prev_conv, prev_kernel, prev_bias)
#         _assign_kernel_and_bias_to_conv_(next_conv, next_kernel)
        
#     # Extend all of the batch norms. Extend scales with 1 padding, shifts with 0 padding
#     for bn in network.batch_norm_enum():
#         new_channels = exp_channels if uniform else bn.weight.data.size(0) * (exp_channels - 1)
        
#         new_scale = _one_pad_1d(bn.weight.data.cpu().numpy(), new_channels)
#         new_shift = _zero_pad_1d(bn.bias.data.cpu().numpy(), new_channels)
#         new_running_mean = _zero_pad_1d(bn.running_mean.data.cpu().numpy(), new_channels)
#         new_running_var = _zero_pad_1d(bn.running_var.data.cpu().numpy(), new_channels)
        
#         _assign_to_batch_norm_(bn, new_scale, new_shift, new_running_mean, new_running_var)
        
        
# def _widen_network_convs_uniformly_(network, exp_channels, init_type='He', function_preserving=True):
#     return _widen_network_convs_(network, exp_channels, init_type, function_preserving, uniform=True)
        
        
# def _widen_network_convs_scaled_(network, exp_channels, init_type='He', function_preserving=True):
#     return _widen_network_convs_(network, exp_channels, init_type, function_preserving, uniform=False)





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
        
        Note: we take the liberty to assume that the shapings of layer are correct to just work.
        """
        if type(layer) is nn.Linear:
            self.fc_extensions.append(layer)
            # TODO: register params
            raise Exception("TODO: register params")
        else:
            self.conv_extensions.append(layer)
            # TODO: register params
            raise Exception("TODO: register params")

        
        
        
        
def make_deeper_network(network, layer):
    """
    Given a network 'network', create a deeper network adding in a new layer 'layer'. 
    
    We assume the our network is build into a conv stack, which feeds into a fully connected stack. 
    """
    if type(network) is not Deepened_Network:
        network = Deepened_Network(network)
    network.extend(layer)
    return network

