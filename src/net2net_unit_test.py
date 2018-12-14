import numpy as np
import torch as t
import torch.nn as nn

from r2r import Mnist_Resnet, Cifar_Resnet, widen_network_
from r2r.net2net import net2net_widen_network_, net2net_make_deeper_network_, Net2Net_conv_identity, HVG, \
    Net2Net_ResBlock_identity
from utils import flatten


def test_function_preserving_net2widernet(model, thresh, function_preserving=True, data_channels=1, verbose=True):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1, data_channels, 32, 32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (scaled) and check that the outputs are (almost) identical
    print("Widening network")
    model = net2net_widen_network_(model, new_channels=4, new_hidden_nodes=2, multiplicative_widen=True)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Average output difference before and after transform is: {val}".format(
            val=t.mean(rand_out - rand_outs[i])))

    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))

    # widen (scaled) and check that the outputs are (almost) identical
    model = net2net_widen_network_(model, new_channels=4, new_hidden_nodes=2, multiplicative_widen=True)

    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")


def test_function_preserving_net2deepernet(model, thresh, data_channels=1, layer1=None, layer2=None, verbose=True):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1, data_channels, 32, 32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # deepen and check that the outputs are (almost) identical

    batch = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(100, data_channels, 32, 32)))
    batch_out = model(batch)
    model = net2net_make_deeper_network_(model, layer1, batch)
    batch_out_new = model(batch)
    err = t.mean(batch_out_new - batch_out)
    if verbose:
        print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
    if t.abs(err) > thresh:
        raise Exception("Unit test failed.")

    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))

    # widen (scaled) and check that the outputs are (almost) identical
    model = net2net_make_deeper_network_(model, layer2, batch)
    batch_out_new = model(batch)
    err = t.mean(batch_out_new - batch_out)
    if verbose:
        print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
    if t.abs(err) > thresh:
        raise Exception("Unit test failed.")


def test_function_preserving_widen_then_deepen(model, thresh, function_preserving=True, data_channels=1, layer=None,
                                               verbose=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    batch = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(100, data_channels, 32, 32)))
    batch_out = model(batch)

    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    model = net2net_widen_network_(model, new_channels=2, new_hidden_nodes=2, multiplicative_widen=True)

    model = net2net_make_deeper_network_(model, layer, batch)
    batch_out_new = model(batch)
    err = t.mean(batch_out_new - batch_out)
    if verbose:
        print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
    if t.abs(err) > thresh:
        raise Exception("Unit test failed.")

    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))


def test_function_preserving_deepen_then_widen(model, thresh, function_preserving=True, data_channels=1, layer=None,
                                               verbose=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    batch = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(100, data_channels, 32, 32)))
    batch_out = model(batch)

    model = net2net_make_deeper_network_(model, layer, batch)
    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    model = net2net_widen_network_(model, new_channels=2, new_hidden_nodes=2, multiplicative_widen=True)

    batch_out_new = model(batch)
    err = t.mean(batch_out_new - batch_out)
    if verbose:
        print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
    if t.abs(err) > thresh:
        raise Exception("Unit test failed.")

    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))


class _Baby_Siamese(nn.Module):
    """
    A small siamese network, with 2 pathways, just to stress test
    """

    def __init__(self):
        super(_Baby_Siamese, self).__init__()
        self.c11 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.c21 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.c22 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.linear1 = nn.Linear((20 + 20) * 32 * 32, 2)
        self.linear2 = nn.Linear(2, 2)

    def conv_forward(self, x):
        x1 = self.c11(x)
        x2 = self.c12(x)
        x = t.cat((x1, x2), 1)
        x1 = self.c21(x)
        x2 = self.c22(x)
        x = t.cat((x1, x2), 1)
        return x

    def fc_forward(self, x):
        x = flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def out_forward(self, x):
        return x

    def forward(self, x):
        x = self.conv_forward(x)
        x = self.fc_forward(x)
        return self.out_forward(x)

    def input_shape(self):
        return (1, 32, 32)

    def conv_hvg(self, cur_hvg):
        root_node = cur_hvg.get_output_nodes()[0]
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c11.weight.size(0) + self.c12.weight.size(0), 32, 32),
                                   input_modules=[self.c11, self.c12],
                                   input_hvns=[root_node, root_node])
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c21.weight.size(0) + self.c22.weight.size(0), 32, 32),
                                   input_modules=[self.c21, self.c22],
                                   input_hvns=[cur_node, cur_node])
        return cur_hvg

    def fc_hvg(self, cur_hvg):
        cur_hvg.add_hvn(hv_shape=(self.linear1.out_features,), input_modules=[self.linear1])
        cur_hvg.add_hvn(hv_shape=(self.linear2.out_features,), input_modules=[self.linear2])
        return cur_hvg

    def hvg(self):
        hvg = HVG(self.input_shape())
        hvg = self.conv_hvg(hvg)
        hvg = self.fc_hvg(hvg)
        return hvg


if __name__ == "__main__":
    verbose = True

    """
    
    if verbose:
        print("\n" * 4)
        print("Testing Net2WiderNet for Mnist Resnet:")
    test_function_preserving_net2widernet(Mnist_Resnet(identity_initialize=False,
                                                       add_residual=False), 1e-5, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2Widernet for Cifar Resnet:")
    test_function_preserving_net2widernet(Cifar_Resnet(identity_initialize=False,
                                                       add_residual=False), 1e-5, data_channels=3, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2DeeperNet for Mnist Resnet:")
    rblock = Net2Net_ResBlock_identity(input_channels=32, intermediate_channels=[32, 32, 32], output_channels=32,
                                       identity_initialize=True, input_spatial_shape=(4, 4))
    rblock2 = Net2Net_ResBlock_identity(input_channels=32, intermediate_channels=[32, 32, 32], output_channels=32,
                                        identity_initialize=True, input_spatial_shape=(4, 4))
    test_function_preserving_net2deepernet(Mnist_Resnet(identity_initialize=False, add_residual=False), 1e-5,
                                           layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2DeeperNet for Cifar Resnet:")
    rblock = Net2Net_ResBlock_identity(input_channels=64, intermediate_channels=[64, 64, 64], output_channels=64,
                                       identity_initialize=True, input_spatial_shape=(4, 4))
    rblock2 = Net2Net_ResBlock_identity(input_channels=64, intermediate_channels=[64, 64, 64], output_channels=64,
                                        identity_initialize=True, input_spatial_shape=(4, 4))
    test_function_preserving_net2deepernet(Cifar_Resnet(identity_initialize=False, add_residual=False), 1e-5,
                                           data_channels=3, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2DeeperNet + Net2WiderNet for Mnist Resnet:")
    rblock = Net2Net_ResBlock_identity(input_channels=32, intermediate_channels=[32, 32, 32], output_channels=32,
                                       identity_initialize=True, input_spatial_shape=(4, 4))
    test_function_preserving_deepen_then_widen(Mnist_Resnet(identity_initialize=False, add_residual=False), 1e-5,
                                               layer=rblock, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2DeeperNet + Net2WiderNet for Cifar Resnet:")
    rblock = Net2Net_ResBlock_identity(input_channels=64, intermediate_channels=[64, 64, 64], output_channels=64,
                                       identity_initialize=True, input_spatial_shape=(4, 4))
    test_function_preserving_deepen_then_widen(Cifar_Resnet(identity_initialize=False, add_residual=False), 1e-5,
                                               data_channels=3, layer=rblock, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2WiderNet + Net2DeeperNet for Mnist Resnet:")
    # insert wider resblock, because widen before deepen this time
    rblock = Net2Net_ResBlock_identity(input_channels=64, intermediate_channels=[64, 64, 64], output_channels=64,
                                       identity_initialize=True, input_spatial_shape=(4, 4))
    test_function_preserving_widen_then_deepen(Mnist_Resnet(identity_initialize=False, add_residual=False), 1e-5,
                                               layer=rblock, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2WiderNet + Net2Deepernet for Cifar Resnet:")
    # insert wider resblock, because widen before deepen this time
    rblock = Net2Net_ResBlock_identity(input_channels=128, intermediate_channels=[128, 128, 128], output_channels=128,
                                       identity_initialize=True, input_spatial_shape=(4, 4))
    test_function_preserving_widen_then_deepen(Cifar_Resnet(identity_initialize=False, add_residual=False), 1e-5,
                                               data_channels=3, layer=rblock, verbose=verbose)

    """
    """
    Testing Baby Siamese
    """

    if verbose:
        print("\n" * 4)
        print("Testing Net2Widernet for siamese network:")
    test_function_preserving_net2widernet(_Baby_Siamese(), 1e-5, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2DeeperNet for siamese network:")
    rblock1 = Net2Net_conv_identity(input_channels=40, kernel_size=(3, 3), input_spatial_shape=(32, 32))
    rblock2 = Net2Net_conv_identity(input_channels=40, kernel_size=(3, 3), input_spatial_shape=(32, 32))
    test_function_preserving_net2deepernet(_Baby_Siamese(), 1e-5, layer1=rblock1, layer2=rblock2, verbose=verbose)


    if verbose:
        print("\n" * 4)
        print("Testing Net2DeeperNet + Net2WiderNet for Siamese Network:")
    rblock = Net2Net_conv_identity(input_channels=40, kernel_size=(3, 3), input_spatial_shape=(32, 32))
    test_function_preserving_deepen_then_widen(_Baby_Siamese(), 1e-5, layer=rblock, verbose=verbose)

    if verbose:
        print("\n" * 4)
        print("Testing Net2WiderNet + Net2DeeperNet for Siamese Network:")
    rblock = Net2Net_conv_identity(input_channels=80, kernel_size=(3, 3), input_spatial_shape=(32, 32))
    test_function_preserving_widen_then_deepen(_Baby_Siamese(), 1e-5, layer=rblock, verbose=verbose)

