import numpy as np



"""
init_utils is where the actual core logic all is. This contains most things implemented at the numpy level.
"""




# All these functions need to be used across r2r. However, they're not exported from the package
__all__ = ['_conv_xavier_initialize', 
           '_conv_he_initialize', 
           '_extend_filter_with_repeated_in_channels', 
           '_extend_filter_with_repeated_out_channels', 
           '_extend_matrix_with_repeated_in_weights', 
           '_extend_matrix_with_repeated_out_weights',
           '_extend_filter_in_channels',
           '_extend_filter_out_channels',
           '_extend_matrix_in_weights',
           '_extend_matrix_out_weights',
           '_zero_pad_1d', 
           '_one_pad_1d',
           '_mean_pad_1d',
           '_widen_hidden_volume']





def _conv_xavier_initialize(filter_shape, override_input_channels=None, override_output_channels=None):
    """
    Initialize a convolutional filter, with shape 'filter_shape', according to "He initialization".
    The weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of 
    sqrt(2/(n_in + n_out)).
    
    This is the initialization of choice for layers with non ReLU activations.
    
    The filter shape should be [output_channels, input_channels, width, height]. So here, 
    n_in = width * height * input_channels and n_out = width * height * output_channels.
    
    When "widening" an filter, from C1 output filters, to C1 + 2*C2 filters, then we want to initialize the 
    additional 2*C2 layers, as if there are C1+2*C2 filters in the output, and therefore we provide the 
    option to override the number of output filters.
    
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
    filter_shape = (out_channels, in_channels, width, height)    
    
    scale = np.sqrt(2.0 / (width*height(in_channels + out_chanels)))
    return scale * np.random.randn(*filter_shape).astype(np.float32) 
    
    
    
    

def _conv_he_initialize(filter_shape, override_input_channels=None):
    """
    Initialize a convolutional filter, with shape 'filter_shape', according to "Xavier initialization".
    Each weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of 
    sqrt(2/n_in).
    
    This is the initialization of choice for layers with ReLU activations.
    
    The filter shape should be [output_channels, input_channels, width, height]. So here, 
    n_in = width * height * input_channels.
    
    As the initization only depends on the number of inputs (the number of input channels), unlike Xavier 
    initialization, we don't need to be able to override the number of output_channels.
    
    :param filter_shape: THe shape of the filter that we want to produce an initialization 
    :param override_output_channels: Override for the number of input filters in the filter_shape (optional)
    :param override_output_channels: unused
    :return: A numpy array, of shape 'filter_shape', randomly initialized according to He initialization.
    """
    out_channels, in_channels, width, height = filter_shape
    if override_input_channels is not None:
        in_channels = override_input_channels
    filter_shape = (out_channels, in_channels, width, height)    
                    
    scale = np.sqrt(2.0 / (in_channels*width*height))
    return scale * np.random.randn(*filter_shape).astype(np.float32)





def _conv_match_scale_initialize(filter_shape, scale=1.0):
    """
    Initialize a convolutional filter, with shape 'filter_shape', with a std deviation of std.
    Each weight for each hidden unit should be drawn from a normal distribution, with zero mean and stddev of
    sqrt(2/n_in).

    :param filter_shape: THe shape of the filter that we want to produce an initialization
    :param scale: The standard deviation of the noise to add
    :return: A numpy array, of shape 'filter_shape', randomly initialized according to He initialization.
    """
    return scale * np.random.randn(*filter_shape).astype(np.float32)




def _extend_filter_with_repeated_out_channels(extending_filter_shape, existing_filter=None, init_type='He', alpha=1.0):
    """
    We want to extend filter by adding output channels with appropriately initialized weights.
    
    Let F be the 'existing_filter', with shape [C1,I,H,W]. 'extending_filter_shape' if the shape 
    by which we want to extend F. 
    
    Let 'extending_filter_shape' be [2*C2,I,H,W]. If the shape value that should be 2*C2 is odd or 
    non-positive, then it's an error.
    
    The shape of the extended filter is [C1+2*C2, I, H, W].
    
    The method initializes E of shape [C2, I, H, W] with the specified 'init_type'. The function returns 
    the concatenation [F;E;alpha*E], where concatination is in the inconsistent dimension and alpha is a 
    multiplicative scalar.
    
    To make a fresh/new filter, with repeated weights, let 'existing_filter' be None, and it will 
    return just [E;alpha*E], as F is "empty". (Concatenation in the output channels dimension.)
    
    :param extending_filter_shape: The shape of the new portion of the filter to return. I.e. [2*C2,I,H,W]
    :param existing_filter: If not None, it must have shape [C1,I,H,W]. This is the existing filter.
    :param init_type: The type of initialization to use for new weights.
    :return: A filter of shape [C1+2*C2, I, H, W], which is the 'existing_filter' extended by 2*C2 output channels. 
            I.e. the filter [F;E;alpha*E]
    """
    # Unpack params input.
    twoC2, I, H, W = extending_filter_shape
    C2 = twoC2 // 2
    C1 = 0 if existing_filter is None else existing_filter.shape[0]
    
    # Error checking.
    if twoC2 % 2 != 0:
        raise Exception("The filter shape needs to be even to allow for repetition in it.")
    elif existing_filter is not None and (H != existing_filter.shape[2]
        or W != existing_filter.shape[3] or I != existing_filter.shape[1]):
        raise Exception("Dimensions of 'extending_filter_shape' and 'existing_filter' are incompatible.")
    
    # Canvas for the new numpy array to return. Copy existing filter weights.
    canvas = np.zeros((C1+twoC2, I, H, W)).astype(np.float32)
    if existing_filter is not None:
        canvas[:C1,:,:,:] = existing_filter

    # Initialize the new weights, and copy that into the canvas (twice).
    new_channels_weights = None
    if init_type == 'He':
        new_channels_weights = _conv_he_initialize((C2,I,H,W))
    elif init_type == 'Xavier':
        new_channels_weights = _conv_xavier_initialize((C2,I,H,W), override_output_channels=C1+twoC2)
    elif init_type == 'match_std':
        scale = np.std(existing_filter) / np.prod([C2,I,H,W])
        new_channels_weights = _conv_match_scale_initialize((C2,I,H,W), scale=scale)
    else:
        raise Exception("Invalid initialization type specified. Please use 'He' or 'Xavier'.")
    
    canvas[C1:C1+C2,:,:,:] = new_channels_weights
    canvas[C1+C2:C1+twoC2,:,:,:] = alpha * new_channels_weights

    # Done :)
    return canvas




def _extend_filter_out_channels(extending_filter_shape, existing_filter=None, init_type='He', alpha=1.0):
    """
    We want to extend filter by adding output channels.

    :param extending_filter_shape: The shape of the new portion of the filter to return. I.e. [2*C2,I,H,W]
    :param existing_filter: If not None, it must have shape [C1,I,H,W]. This is the existing filter.
    :param init_type: The type of initialization to use for new weights.
    :return: A filter of shape [C1+2*C2, I, H, W], which is the 'existing_filter' extended by 2*C2 output channels.
            I.e. the filter [F;E;alpha*E]
    """
    # Unpack params input.
    twoC2, I, H, W = extending_filter_shape
    C1 = 0 if existing_filter is None else existing_filter.shape[0]

    # Error checking.
    if existing_filter is not None and (H != existing_filter.shape[2]
        or W != existing_filter.shape[3] or I != existing_filter.shape[1]):
        raise Exception("Dimensions of 'extending_filter_shape' and 'existing_filter' are incompatible.")

    # Canvas for the new numpy array to return. Copy existing filter weights.
    canvas = np.zeros((C1+twoC2, I, H, W)).astype(np.float32)
    if existing_filter is not None:
        canvas[:C1,:,:,:] = existing_filter

    # If init with zero's, then we're done
    if init_type == 'zero':
        return canvas

    # Initialize the new weights, and copy that into the canvas (twice).
    new_channels_weights = None
    if init_type == 'He':
        new_channels_weights = _conv_he_initialize((twoC2,I,H,W))
    elif init_type == 'Xavier':
        new_channels_weights = _conv_xavier_initialize((twoC2,I,H,W), override_output_channels=C1+twoC2)
    elif init_type == 'match_std':
        scale = np.std(existing_filter) / np.prod([twoC2,I,H,W])
        new_channels_weights = _conv_match_scale_initialize((twoC2,I,H,W), scale=scale)
    else:
        raise Exception("Invalid initialization type specified. Please use 'He' or 'Xavier'.")

    canvas[C1:C1+twoC2,:,:,:] = new_channels_weights

    # Done :)
    return canvas





def _extend_filter_with_repeated_in_channels(extending_filter_shape, existing_filter=None, init_type='He', alpha=1.0):
    """
    We want to extend filter by adding input channels with appropriately initialized weights.
    
    This time we are repeating/concatenating over the input channels rather than output channels. 
    They're nearly identical. 
    (There is/was code to extend both input and output channels at once, but it's overly complex to use here.)
    
    Let F be the 'existing_filter', with shape [C,I1,H,W]. 'extending_filter_shape' if the shape 
    by which we want to extend F. 
    
    Let 'extending_filter_shape' be [C,2*I2,H,W]. If the shape value that should be '2*I2' is odd or 
    non-positive, then it's an error.
    
    The shape of the extended filter is [C, I1+2*I2, H, W].
    
    The method initializes E of shape [C, I2, H, W] with the specified 'init_type'. The function returns 
    the concatenation [F;E;alpha*E], where concatination is in the inconsistent dimension and alpha is a 
    multiplicative scalar.
    
    To make a fresh/new filter, with repeated weights, let 'existing_filter' be None, and it will 
    return just [E;alpha*E], as F is "empty". (Concatenation in the input channel dimension.)
    
    :param extending_filter_shape: The shape of the new portion of the filter to return. I.e. [C,2*I2,H,W]
    :param existing_filter: If not None, it must have shape [C,I1,H,W]. This is the existing filter.
    :param init_type: The type of initialization to use for new weights.
    :return: A filter of shape [C, I1+2*I2, H, W], which is the 'existing_filter' extended by 2*I2 input channels. 
            I.e. the filter [F;E;alpha*E]
    """
    # Unpack params input.
    C, twoI2, H, W = extending_filter_shape
    I2 = twoI2 // 2
    I1 = 0 if existing_filter is None else existing_filter.shape[1]
    
    # Error checking.
    if twoI2 % 2 != 0:
        raise Exception("The filter shape needs to be even to allow for repetition in it.")
    elif existing_filter is not None and (H != existing_filter.shape[2]
        or W != existing_filter.shape[3] or C != existing_filter.shape[0]):
        raise Exception("Dimensions of 'extending_filter_shape' and 'existing_filter' are incompatible.")
    
    # Canvas for the new numpy array to return. Copy existing filter weights.
    canvas = np.zeros((C, I1+twoI2, H, W)).astype(np.float32)
    if existing_filter is not None:
        canvas[:,:I1,:,:] = existing_filter

    # Initialize the new weights, and copy that into the canvas (twice).
    new_channels_weights = None
    if init_type == 'He':
        new_channels_weights = _conv_he_initialize((C,I2,H,W))
    elif init_type == 'Xavier':
        new_channels_weights = _conv_xavier_initialize((C,I2,H,W), override_input_channels=I1+twoI2)
    elif init_type == 'match_std':
        scale = np.std(existing_filter) / np.prod([C,I2,H,W])
        new_channels_weights = _conv_match_scale_initialize((C,I2,H,W), scale=scale)
    else:
        raise Exception("Invalid initialization type specified. Please use 'He' or 'Xavier'.")
    
    canvas[:,I1:I1+I2,:,:] = new_channels_weights
    canvas[:,I1+I2:I1+twoI2,:,:] = alpha * new_channels_weights

    # Done :)
    return canvas





def _extend_filter_in_channels(extending_filter_shape, existing_filter=None, init_type='He', alpha=1.0):
    """
    We want to extend filter by adding input channels

    :param extending_filter_shape: The shape of the new portion of the filter to return. I.e. [C,2*I2,H,W]
    :param existing_filter: If not None, it must have shape [C,I1,H,W]. This is the existing filter.
    :param init_type: The type of initialization to use for new weights.
    :return: A filter of shape [C, I1+2*I2, H, W], which is the 'existing_filter' extended by 2*I2 input channels.
            I.e. the filter [F;E;alpha*E]
    """
    # Unpack params input.
    C, twoI2, H, W = extending_filter_shape
    I1 = 0 if existing_filter is None else existing_filter.shape[1]

    # Error checking.
    if existing_filter is not None and (H != existing_filter.shape[2]
        or W != existing_filter.shape[3] or C != existing_filter.shape[0]):
        raise Exception("Dimensions of 'extending_filter_shape' and 'existing_filter' are incompatible.")

    # Canvas for the new numpy array to return. Copy existing filter weights.
    canvas = np.zeros((C, I1+twoI2, H, W)).astype(np.float32)
    if existing_filter is not None:
        canvas[:,:I1,:,:] = existing_filter

    # If init with zero's, then we're done
    if init_type == 'zero':
        return canvas

    # Initialize the new weights, and copy that into the canvas (twice).
    new_channels_weights = None
    if init_type == 'He':
        new_channels_weights = _conv_he_initialize((C,twoI2,H,W))
    elif init_type == 'Xavier':
        new_channels_weights = _conv_xavier_initialize((C,twoI2,H,W), override_input_channels=I1+twoI2)
    elif init_type == 'match_std':
        scale = np.std(existing_filter) / np.prod([C,twoI2,H,W])
        new_channels_weights = _conv_match_scale_initialize((C,twoI2,H,W), scale=scale)
    else:
        raise Exception("Invalid initialization type specified. Please use 'He' or 'Xavier'.")

    canvas[:,I1:I1+twoI2,:,:] = new_channels_weights

    # Done :)
    return canvas
    

    
    
    
def _extend_matrix_with_repeated_out_weights(extending_matrix_shape, existing_matrix=None, init_type='He', alpha=1.0):
    """
    Re-implement _extend_filter_with_repeated_out_channels, but for fully connected layers.
    We can reshape our inputs to just re-use _extend_filter_with_repeated_out_channels.
    
    :param extending_matrix_shape: The shape of the matrix for which we wish to to the existing matrix
    :param existing_matrix: The existing weights of a linear layer (can be None to initialize a new matrix with
            duplicated weights)
    :param init_type: The type of initialization to use for the new weights.
    :param alpha: The coefficient of the repeated weights
    :return: The new, widened matrix for the fully connected layer
    """
    return _extend_matrix_helper(_extend_filter_with_repeated_out_channels, extending_matrix_shape, existing_matrix,
                                 init_type, alpha)





def _extend_matrix_out_weights(extending_matrix_shape, existing_matrix=None, init_type='He'):
    """
    Re-implement _extend_filter_out_channels, but for fully connected layers.
    We can reshape our inputs to just re-use _extend_filter_out_channels.

    :param extending_matrix_shape: The shape of the matrix for which we wish to to the existing matrix
    :param existing_matrix: The existing weights of a linear layer (can be None to initialize a new matrix with
            duplicated weights)
    :param init_type: The type of initialization to use for the new weights.
    :return: The new, widened matrix for the fully connected layer
    """
    return _extend_matrix_helper(_extend_filter_out_channels, extending_matrix_shape, existing_matrix, init_type, 1.0)
    
    
    
    

def _extend_matrix_with_repeated_in_weights(extending_matrix_shape, existing_matrix=None, init_type='He', alpha=1.0):
    """
    Re-implement _extend_filter_with_repeated_in_channels, but for fully connected layers.
    We can reshape our inputs to just re-use _extend_filter_with_repeated_in_channels.
    
    :param extending_matrix_shape: The shape of the matrix for which we wish to to the existing matrix
    :param existing_matrix: The existing weights of a linear layer (can be None to initialize a new matrix with
            duplicated weights)
    :param init_type: The type of initialization to use for the new weights.
    :param alpha: The coefficient of the repeated weights
    :return: The new, widened matrix for the fully connected layer
    """
    return _extend_matrix_helper(_extend_filter_with_repeated_in_channels, extending_matrix_shape, existing_matrix,
                                 init_type, alpha)





def _extend_matrix_in_weights(extending_matrix_shape, existing_matrix=None, init_type='He'):
    """
    Re-implement _extend_filter_in_channels, but for fully connected layers.
    We can reshape our inputs to just re-use _extend_filter_in_channels.

    :param extending_matrix_shape: The shape of the matrix for which we wish to to the existing matrix
    :param existing_matrix: The existing weights of a linear layer (can be None to initialize a new matrix with
            duplicated weights)
    :param init_type: The type of initialization to use for the new weights.
    :return: The new, widened matrix for the fully connected layer
    """
    return _extend_matrix_helper(_extend_filter_in_channels, extending_matrix_shape, existing_matrix, init_type, 1.0)





def _extend_matrix_helper(extend_fn, extending_matrix_shape, existing_matrix=None, init_type='He', alpha=1.0):
    """
    _extend_matrix_with_repeated_out_weights and _extend_matrix_with_repeated_in_weights have exactly the same
    structure, except one calls _extend_filter_with_repeated_out_channels recursively, and the other 
    calls _extend_filter_with_repeated_in_channels recursively.
    
    extend_fn is appropriately set to _extend_filter_with_repeated_out_channels or 
    _extend_filter_with_repeated_in_channels
    """
    n_out, n_in = extending_matrix_shape
    psuedo_extending_filter_shape = (n_out,n_in,1,1)
    
    psuedo_existing_filter = existing_matrix
    if psuedo_existing_filter is not None:
        psuedo_existing_filter = np.expand_dims(np.expand_dims(psuedo_existing_filter, axis=2), axis=3)
    
    widened_psuedo_filter = extend_fn(psuedo_extending_filter_shape, psuedo_existing_filter, init_type, alpha)
    
    return np.squeeze(widened_psuedo_filter)





def _zero_pad_1d(old_val, new_params):
    """
    Zero pads an old (1d) tensor to match the new number of outputs
    
    :param old_val the old numpy tensor to zero pad
    :param new_params: the number of new params needed
    :return: a new, padded tensor
    """
    old_len = old_val.shape[0]
    canvas = np.zeros((old_len+new_params,), dtype=np.float32)
    canvas[:old_len] = old_val
    return canvas





def _one_pad_1d(old_val, new_params):
    """
    One pads an old (1d) tensor to match the new number of outputs
    
    :param old_val the old numpy tensor to one pad
    :param new_params: the number of new params needed
    :return: a new, padded tensor
    """
    old_len = old_val.shape[0]
    canvas = np.ones((old_len+new_params,), dtype=np.float32)
    canvas[:old_len] = old_val
    return canvas





def _mean_pad_1d(old_val, new_params, ratio=1.0):
    """
    Pads an old (1d) tensor to match the new number of outputs. Pads with the mean value of 'old_val'

    :param old_val the old numpy tensor to one pad
    :param new_params: the number of new params needed
    :param ratio: a ratio to use for the mean value
    :return: a new, padded tensor
    """
    old_len = old_val.shape[0]
    canvas = np.ones((old_len+new_params,), dtype=np.float32) * np.mean(old_val) * ratio
    canvas[:old_len] = old_val
    return canvas





def _widen_hidden_volume(prev_kernel, prev_bias, next_kernel, extra_channels=0, init_type='He', 
                         function_preserving=True):
    """
    This is an implementation of our "R2WiderR". 
    
    This function will widen a single hidden volume as part of a CNN. To widen the volume, we need to 
    add additional output filters to the kernel before this volume. Then, we also need to add additonal 
    input filters to the kernel after this volume. 
    
    Having written '_extend_filter_with_repeated_out_channels' and '_extend_filter_with_repeated_in_channels'
    very generally, we can piggyback off that prior work. If we want the widening to be function preserving 
    then set alpha=-1.0 in the repeated in channels of the next kernel, to implement the R2WiderR transform,
    as described in the paper.
    
    :param prev_kernel: The numpy Tensor for the kernel of the convolution that computes the hidden 
            volume that we are extending. (I.e. the kernel preceeding the volume being extended.) 
    :param prev_bias: The numpy Tensor for the bias of the convolution preceeding the volume being extended.
    :param next_kernel: The numpy Tensor for the kernel of the convolution used after this volume.
    :param extra_channels: The number of extra channels to add to the hidden volume.
    :param init_type: The tpye
    :return: new_prev_kernel, new_prev_bias, new_next_kernel. Which are all numpy Tensors to be appropriately 
            set allocated as PyTorch Parameters in the networks.
    """
    # For us to be able to perform a function preserving transforms, extra channels must be even and non-negative
    if extra_channels <= 0 or (function_preserving and extra_channels % 2 != 0):
        raise Exception("Invalid number of extra channels in widen.")
        
    # Compute new kernel for the 'prev_kernel'
    _, in_channels, width, height = prev_kernel.shape
    kernel_extra_shape = (extra_channels, in_channels, width, height)
    prev_kernel = _extend_filter_with_repeated_out_channels(kernel_extra_shape, prev_kernel, init_type)
    
    # Zero pad the 'prev_bias'
    prev_bias = _zero_pad_1d(prev_bias, extra_channels)
    
    # Compute the new kernel for 'next_kernel'
    alpha = -1.0 if function_preserving else 1.0
    out_channels, _, width, height = next_kernel.shape 
    kernel_extra_shape = (out_channels, extra_channels, width, height)
    next_kernel = _extend_filter_with_repeated_in_channels(kernel_extra_shape, next_kernel, init_type, alpha)
    
    # Doneee :)
    return prev_kernel, prev_bias, next_kernel
