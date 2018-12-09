import numpy as np
import torch as t
import torch.nn as nn

from r2r import Mnist_Resnet, Cifar_Resnet, widen_network_
from r2r.net2net import net2net_widen_network_, net2net_make_deeper_network_, Net2Net_conv_identity, HVG
from utils import flatten


def test_function_preserving_net2widernet(model, thresh, function_preserving=True, data_channels=1):
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
    print ("Widening network")
    model = net2net_widen_network_(model, new_channels=4, new_hidden_nodes=2, scaled=True)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Average output difference before and after transform is: {val}".format(
            val=t.mean(rand_out - rand_outs[i])))

    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    print("Params before the transform is: {param}".format(param=params_before))
    print("Params after the transform is: {param}".format(param=params_after))

    # widen (scaled) and check that the outputs are (almost) identical
    model = net2net_widen_network_(model, new_channels=4, new_hidden_nodes=2, scaled=True)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Avg output difference before and after ANOTHER transform is: {val}".format(
            val=t.mean(rand_out - rand_outs[i])))


def test_function_preserving_net2deepernet(model, thresh, data_channels=1, layer1=None,
                                       layer2=None):
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

    batch = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(500, data_channels, 32, 32)))
    batch_out = model(batch)
    model = net2net_make_deeper_network_(model, layer1, batch)
    batch_out_new = model(batch)
    print("BATCH Average output difference before and after transform is: {val}".format(
        val=t.mean(batch_out_new - batch_out)))

    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Average output difference before and after transform is: {val}".format(
            val=t.mean(rand_out - rand_outs[i])))

    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    print("Params before the transform is: {param}".format(param=params_before))
    print("Params after the transform is: {param}".format(param=params_after))

    # widen (scaled) and check that the outputs are (almost) identical
    model = net2net_make_deeper_network_(model, layer2, batch)

    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Avg output difference before and after ANOTHER transform is: {val}".format(
            val=t.mean(rand_out - rand_outs[i])))


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
        self.c31 = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.linear1 = nn.Linear((20 + 20) * 32 * 32, 2)
        self.linear2 = nn.Linear(2, 2)

    def conv_forward(self, x):
        x1 = self.c11(x)
        x2 = self.c12(x)
        x = t.cat((x1, x2), 1)
        x1 = self.c21(x)
        x2 = self.c22(x)
        x = t.cat((x1, x2), 1)
        x = self.c31(x)
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

    def lle_or_hvg(self):
        return "hvg"

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
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c31.weight.size(0), 32, 32),
                                   input_modules=[self.c31])
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
    print("Testing R2WiderR for Mnist Resnet:")
    test_function_preserving_net2widernet(Mnist_Resnet(add_residual=False), 0.0001)

    print("\n"*4)
    print("Testing R2DeeperR for Mnist Resnet:")
    model = _Baby_Siamese()
    rblock = Net2Net_conv_identity(input_channels=40, kernel_size=(3, 3), input_spatial_shape=(32,32))
    rblock2 = Net2Net_conv_identity(input_channels=40, kernel_size=(3, 3), input_spatial_shape=(32,32))
    test_function_preserving_net2deepernet(model, 0.0001, layer1=rblock, layer2=rblock2)

    print("\n" * 4)
    print("Testing R2WiderR for siamese network:")
    test_function_preserving_net2widernet(_Baby_Siamese(), 0.0001)
