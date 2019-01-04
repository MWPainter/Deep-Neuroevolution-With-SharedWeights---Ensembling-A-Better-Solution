"""
Script to identify what's wrong with batch norm at the moment, or, a lack of knowledge about how BatchNorm functions.
"""
import torch as t
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

# Create a batch norm + run a forward pass
rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(100, 32, 64, 64)))
bn = nn.BatchNorm2d(num_features=32, momentum=1.0)
bn(rand_in)

# Compare variance of input to running var maintained
np_var = np.var(rand_in.numpy(), axis=(0,2,3))
bn_var = bn.running_var.numpy()
print(np.max(np.abs(np_var - bn_var)))

# Compare the means
np_mean = np.mean(rand_in.numpy(), axis=(0,2,3))
bn_mean = bn.running_mean.numpy()
print(np.max(np.abs(np_mean - bn_mean)))

# Try setting bn to be an identity
bn.weight = Parameter(t.tensor(np.sqrt(bn_var + bn.eps)))
bn.bias = Parameter(t.tensor(bn.running_mean.numpy()))
bn.register_parameter('weight', bn.weight)
bn.register_parameter('bias', bn.bias)
bn.running_mean = t.tensor(bn.running_mean.numpy())
bn.running_var = t.tensor(bn.running_var.numpy())

# Test difference betwen out and in
out = bn(rand_in)
print(np.max(np.abs(rand_in.detach().numpy() - out.detach().numpy())))

# Difference between two sequential runs through batch norm
out1 = bn(rand_in)
out2 = bn(rand_in)
print(np.max(np.abs(out1.detach().numpy() - out2.detach().numpy())))