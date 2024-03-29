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
    :return: A reference to the updated conv module
    """
    # Check if module actually has a bias.
    module_has_bias = conv.bias is not None

    # If bias is none, and module has a bias, we can safely re-assign the old bias
    if module_has_bias and bias is None:
        bias = conv.bias.data.cpu().numpy()
    
    # Sanity check new shaping values
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape
    if module_has_bias:
        bias_size = bias.shape[0]
        if bias_size != out_channels:
            raise Exception("Trying to assign inconsistend kernel and biases in convolution.")
    _, _, old_kernel_height, old_kernel_width = conv.weight.data.size()
    if old_kernel_height != kernel_height or old_kernel_width != kernel_width:
        raise Exception("Old kernel and new kernel (spatial) shapes are inconsistent in conv assignment.")
    
    # Update weight, bias and shape values in 'conv'
    conv.weight = Parameter(t.tensor(kernel))
    if module_has_bias:
        conv.bias = Parameter(t.tensor(bias))
    
    conv.out_channels = out_channels
    conv.in_channels = in_channels
    conv.kernel_size = (kernel_width, kernel_height)
    
    # Re-register the new weight and bias in the conv
    conv.register_parameter('weight', conv.weight)
    if module_has_bias:
        conv.register_parameter('bias', conv.bias)
    
    # Move to GPU if need be
    conv = cudafy(conv)
        
    return conv





def _assign_to_batch_norm_(batch_norm, scale, bias, run_mean, run_var):
    """
    Update 'batch_norm' with a new scale, bias and running mean and running variance

    :param batch_norm: An instance of nn.BatchNorm to assign the scale/bias/mean/var to
    :param scale: A numpy tensor to assign to nn.BatchNorm.weight (the scale parameter of batch norm)
    :param bias: A numpy tensor to assign to nn.BatchNorm.bias (the shift parameter of batch norm)
    :param run_mean: A numpy tensor to assign to nn.BatchNorm.running_mean (the exp running mean of batch norm)
    :param run_var: A numpy tensor to assign to nn.BatchNorm.running_var (the exp running var of batch norm)
    :return: A reference to the updated batch norm module.
    """
    # Sanity checks
    l = scale.shape[0]
    if bias.shape[0] != l or run_mean.shape[0] != l or run_var.shape[0] != l:
        raise Exception("Attempting to assign tensors with inconsistent shaping in batch norm.")

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
    :return: A reference to the updated linear module
    """
    # Check if module actually has a bias.
    module_has_bias = linear.bias is not None

    # If bias is none, and module has a bias, we can safely re-assign the old bias
    if module_has_bias and bias is None:
        bias = linear.bias.data.cpu().numpy()
    
    # Sanity check new shaping values
    n_out, n_in = matrix.shape
    if module_has_bias:
        bias_size = bias.shape[0]
        if bias_size != n_out:
            raise Exception("Trying to assign inconsistend matrix and biases in linear.")
    
    # Update weight, bias and shape values in 'conv'
    linear.weight = Parameter(t.tensor(matrix))
    if module_has_bias:
        linear.bias = Parameter(t.tensor(bias))
    
    linear.out_features = n_out
    linear.in_features = n_in
    
    # Re-register the new weight and bias in the conv
    linear.register_parameter('weight', linear.weight)
    if module_has_bias:
        linear.register_parameter('bias', linear.bias)
    
    # Move to GPU if need be
    linear = cudafy(linear)
        
    return linear
