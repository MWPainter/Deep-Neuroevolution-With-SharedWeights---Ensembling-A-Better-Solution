import numpy as np
import torch as t
import torch.nn as nn

from r2r import Res_Block, Mnist_Resnet, Cifar_Resnet, widen_network_, make_deeper_network_, HVG
from utils import flatten


    
    
    
def test_function_preserving_r2deeperr(model, thresh, function_preserving=True, data_channels=1, layer1=None,
                                       layer2=None, verbose=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1,data_channels,32,32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # deepen and check that the outputs are (almost) identical
    model = make_deeper_network_(model, layer1)
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if err > thresh:
            raise Exception("Unit test failed.")
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))

    # widen (scaled) and check that the outputs are (almost) identical
    model = make_deeper_network_(model, layer2)
    
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
        if err > thresh:
            raise Exception("Unit test failed.")



def test_function_preserving_r2widerr(model, thresh, function_preserving=True, data_channels=1, verbose=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1,data_channels,32,32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (scaled) and check that the outputs are (almost) identical
    model = widen_network_(model, new_channels=4, new_hidden_nodes=2, init_type='He', 
                           function_preserving=function_preserving, scaled=True)
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if err > thresh:
            raise Exception("Unit test failed.")
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))

    # widen (scaled) and check that the outputs are (almost) identical
    model = widen_network_(model, new_channels=4, new_hidden_nodes=2, init_type='He', 
                           function_preserving=function_preserving, scaled=True)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
        if err > thresh:
            raise Exception("Unit test failed.")
        
        
        
def test_function_preserving_deepen_then_widen(model, thresh, function_preserving=True, data_channels=1, layer=None,
                                               verbose=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1,data_channels,32,32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (scaled) and check that the outputs are (almost) identical
    model = make_deeper_network_(model, layer)
    model = widen_network_(model, new_channels=2, new_hidden_nodes=0, init_type='He', 
                           function_preserving=function_preserving, scaled=True)
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if err > thresh:
            raise Exception("Unit test failed.")
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))
        
        
def test_function_preserving_widen_then_deepen(model, thresh, function_preserving=True, data_channels=1, layer=None,
                                               verbose=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1,data_channels,32,32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (scaled) and check that the outputs are (almost) identical
    model = widen_network_(model, new_channels=2, new_hidden_nodes=2, init_type='He', 
                           function_preserving=function_preserving, scaled=True)
    model = make_deeper_network_(model, layer)
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(rand_out - rand_outs[i])
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if err > thresh:
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
        self.c31 = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.linear1 = nn.Linear((20+20)*32*32, 2)
        self.linear2 = nn.Linear(2, 2)
        
    def conv_forward(self, x):
        x1 = self.c11(x)
        x2 = self.c12(x)
        x = t.cat((x1,x2), 1)
        x1 = self.c21(x)
        x2 = self.c22(x)
        x = t.cat((x1,x2), 1)
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
        return (1,32,32)
    
    def conv_hvg(self, cur_hvg):
        root_node = cur_hvg.get_output_nodes()[0]
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c11.weight.size(0)+self.c12.weight.size(0), 32, 32),
                                   input_modules=[self.c11, self.c12], 
                                   input_hvns=[root_node, root_node])
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c21.weight.size(0)+self.c22.weight.size(0), 32, 32), 
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
    verbose = False

    if verbose:
        print("Testing R2WiderR for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet(), 1e-5, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet(), 1e5, False, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet(), 1e-5, data_channels=3, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet(), 1e5, False, data_channels=3, verbose=verbose)
    
    

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for Mnist Resnet:")
    rblock = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Mnist_Resnet(), 1e-5, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for Mnist Resnet:")
    rblock = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=False, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=False, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Mnist_Resnet(), 1e5, False, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Cifar_Resnet(), 1e-5, data_channels=3, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=False, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=False, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Cifar_Resnet(), 1e5, False, data_channels=3, layer1=rblock, layer2=rblock2, verbose=verbose)
    
    

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR + R2WiderR for Mnist Resnet:")
    rblock = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_deepen_then_widen(Mnist_Resnet(), 1e-5, layer=rblock, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR + R2WiderR for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_deepen_then_widen(Cifar_Resnet(), 1e-5, data_channels=3, layer=rblock, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR + R2DeeperR for Mnist Resnet:")
    # insert wider resblock, because widen before deepen this time
    rblock = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=True, input_spatial_shape=(4,4)) 
    test_function_preserving_widen_then_deepen(Mnist_Resnet(), 1e-5, layer=rblock, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR + R2DeeperR for Cifar Resnet:")
    # insert wider resblock, because widen before deepen this time
    rblock = Res_Block(input_channels=128, intermediate_channels=[2,2,2], output_channels=128, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_widen_then_deepen(Cifar_Resnet(), 1e-5, data_channels=3, layer=rblock, verbose=verbose)
    
    

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese(), 1e-5, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding widening for siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese(), 1e5, False, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for siamese network:")
    rblock1 = Res_Block(input_channels=40, intermediate_channels=[2,2,2], output_channels=40, 
                       identity_initialize=True, input_spatial_shape=(32,32))
    rblock2 = Res_Block(input_channels=40, intermediate_channels=[2,2,2], output_channels=40, 
                       identity_initialize=True, input_spatial_shape=(32,32))
    test_function_preserving_r2deeperr(_Baby_Siamese(), 1e-5, layer1=rblock1, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for siamese network:")
    rblock1 = Res_Block(input_channels=40, intermediate_channels=[2,2,2], output_channels=40, 
                       identity_initialize=False, input_spatial_shape=(32,32))
    rblock2 = Res_Block(input_channels=40, intermediate_channels=[2,2,2], output_channels=40, 
                       identity_initialize=False, input_spatial_shape=(32,32))
    test_function_preserving_r2deeperr(_Baby_Siamese(), 1e5, False, layer1=rblock1, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR + R2WiderR for Siamese Network:")
    rblock = Res_Block(input_channels=40, intermediate_channels=[10,10,10], output_channels=40, 
                       identity_initialize=True, input_spatial_shape=(32,32))
    test_function_preserving_deepen_then_widen(_Baby_Siamese(), 1e-5, layer=rblock, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR + R2DeeperR for Siamese Network:")
    rblock = Res_Block(input_channels=80, intermediate_channels=[10,10,10], output_channels=80, 
                       identity_initialize=True, input_spatial_shape=(32,32))
    test_function_preserving_widen_then_deepen(_Baby_Siamese(), 1e-5, layer=rblock, verbose=verbose)
    
    