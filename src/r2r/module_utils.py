import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter

from utils.pytorch_utils import cudafy





"""
module_utils.py contains all of the code concerning "assignment" of new weights to nn.Module objects.
"""




__all__ = ['_assign_kernel_and_bias_to_conv_', 
           '_assign_to_batch_norm_',
           '_assign_weights_and_bias_to_linear_']





def _assign_kernel_and_bias_to_conv_(conv, kernel, bias=None):
    """
    Assigns 'kernel' and 'bias' to be the kernel and bias of the nn.Conv2d 'conv'.
    If 'bias' is None, then, we keep the old bias.
    
    :param conv: An instance of nn.Conv2d to assign the kernel to
    :param kernel: A numpy tensor to assign to nn.Conv2d.weight (the kernel parameter)
    :return: A reference to the updated conv
    """
    # If bias is none, we can safely re-assign the old bias
    if bias is None:
        bias = conv.bias.data.cpu().numpy()
    
    # Sanity check new shaping values
    out_channels, in_channels, kernel_width, kernel_height = kernel.shape
    bias_size = bias.shape[0]
    if bias_size != out_channels:
        raise Exception("Trying to assign inconsistend kernel and biases in convolution.")
    
    # Update weight, bias and shape values in 'conv'
    conv.weight = Parameter(t.tensor(kernel))
    conv.bias = Parameter(t.tensor(bias))
    
    conv.out_channels = out_channels
    conv.in_channels = in_channels
    conv.kernel_size = (kernel_width, kernel_height)
    
    # Re-register the new weight and bias in the conv
    conv.register_parameter('weight', conv.weight)
    conv.register_parameter('bias', conv.bias)
    
    # Move to GPU if need be
    conv = cudafy(conv)
        
    return conv





def _assign_to_batch_norm_(batch_norm, scale, bias, run_mean, run_var):
    """
    Update 'batch_norm' with a new scale, bias and running mean and running variance
    """
    # Assign new params
    batch_norm.weight = Parameter(t.tensor(scale))
    batch_norm.bias = Parameter(t.tensor(bias))
    
    # Re register new params, and set running mean and var
    batch_norm.register_parameter('weight', batch_norm.weight)
    batch_norm.register_parameter('bias', batch_norm.bias)
    batch_norm.running_mean = t.tensor(run_mean)
    batch_norm.running_var = t.tensor(run_var)
    
    # Move to gpu if need be
    batch_norm = cudafy(batch_norm)
    
    return batch_norm





def _assign_weights_and_bias_to_linear_(linear, matrix, bias=None):
    """
    Given a linear nn.Module, 'linear', we assign 'matrix' to the weights of that linear layer, 
    and if not none, we assign 'bias' to be the bias of 'linear'.
    
    If 'bias' is none, then we assume the current bias (with it's current shape) is safe to use. 
    If we are increasing the number of outputs from the hidden layer, then we must assign a new (probably zero 
    padded) bias.
    
    :param linear: An instance of nn.Linear to assign the kernel to
    :param matrix: A numpy tensor to assign to linear.weight
    :param bias: A numpy tensor to assign to linear.bias.
    :return: A reference to the updated linear layer
    """
    # If bias is none, we assume we can safely re-assign the old bias
    if bias is None:
        bias = linear.bias.data.cpu().numpy()
    
    # Sanity check new shaping values
    n_out, n_in = matrix.shape
    bias_size = bias.shape[0]
    if bias_size != n_out:
        raise Exception("Trying to assign inconsistend matrix and biases in linear.")
    
    # Update weight, bias and shape values in 'conv'
    linear.weight = Parameter(t.tensor(matrix))
    linear.bias = Parameter(t.tensor(bias))
    
    linear.out_features = n_out
    linear.in_features = n_in
    
    # Re-register the new weight and bias in the conv
    linear.register_parameter('weight', linear.weight)
    linear.register_parameter('bias', linear.bias)
    
    # Move to GPU if need be
    linear = cudafy(linear)
        
    return linear
