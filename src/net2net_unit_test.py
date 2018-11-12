import numpy as np
import torch as t
import torch.nn as nn

from r2r import Mnist_Resnet, Cifar_Resnet, widen_network_
from r2r.net2net import net2net_widen_network_


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
    model = widen_network_(model, new_channels=4, new_hidden_nodes=2, scaled=True)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Avg output difference before and after ANOTHER transform is: {val}".format(
            val=t.mean(rand_out - rand_outs[i])))



if __name__ == "__main__":
    print("Testing R2WiderR for Mnist Resnet:")
    #test(Mnist_Resnet())
    test_function_preserving_net2widernet(Mnist_Resnet(), 0.0001)
