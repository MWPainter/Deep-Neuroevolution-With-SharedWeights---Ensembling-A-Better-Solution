import torch as t





"""
Helper functions that really should be built into pytorch
"""




__all__ = ['cudafy', 'flatten']





def cudafy(x):
    """
    If 'x' can be moved to the gpu, then do so.
    """
    if t.cuda.is_available():
        return x.cuda()
    return x



def flatten(x):
    """
    Flattens a tensor, 'x', with shape (batch_size, ...)
    """
    batch_size = x.size(0)
    return x.view((batch_size, -1))