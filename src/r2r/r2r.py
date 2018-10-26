

"""
This file contains all of the logic related to performing widening and deepening transforms and initializations.

For a nn.Module to be widened it needs to implement the following:
network_instance.


We use the following python and PyTorch conventions for function naming very explicitly:
_function_name = a private/helper function, not to be exported;
function_name_ = a function that has side effects (will change the objects that have been passed into them).

Note, that this means that we will sometimes be declaring function names like '_function_name_', when both cases are true.

"""

"""
TODO: implement this
"""    
# Only export things that actually widen/deepen volumes
# all = []




"""
TODO: implement this
"""    
# class Volume_Node(object):
#     """
#     A node containing all of the information needed to widen a specific volume. 
#     For any widening, we must form a graph over the network consisting of these volume nodes.
    
#     We keep track of the following:
#     - parent nodes (explicitly specified in initializer)
#     - shape (explicitly specified in initializer)
#     - any batch norm layer associated with it (explicitly specified in initializer)
#     - if this volume has yet to be widened 
#     - input dimension
#     - output dimension (output dimension != input dimension if input is from convs and feeding it into linear layers)
#     """
#     def __init__(self):
#         pass

    
"""
TODO: implement this
"""    
# class Volume_Graph_Enumerator(object):
#     """
#     Iterates over the graph contents to provide appropriate tuples used for widening transforms.
#     """
#     def __inti__(self):
#         pass




# R2WiderR_conv
# R2DeeperR_conv
# R2WiderR_fc (reshape and use R2WiderR_conv)
# R2DeeperR_fc (reshape and use R2DeeperR_conv)

# widen helper -> widen all layers in a network
# widen_helper_conv_only -> widens conv layers only + corresponding bn layers
# widen_helper_fc_only -> widens fc layers only + corresponding bn layers