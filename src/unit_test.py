import numpy as np
import torch as t
import torch.nn as nn

from r2r import Res_Block, Mnist_Resnet, Cifar_Resnet, widen_network_, make_deeper_network_, HVG, InceptionV4, resnet50, resnet18
from utils import flatten, parameter_magnitude



def test_res_block_identity_initialize(thresh=1.0e-5):
    rblock = Res_Block(input_channels=4, intermediate_channels=[4,4,4], output_channels=4,
                       identity_initialize=True, input_spatial_shape=(4,4))
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-0.5, high=0.5, size=(1,4,4,4)))
        rand_out = rblock(rand_in)
        err = t.mean(rand_out - rand_in)
        if verbose:
            print("Average difference between input and output of res block is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")




def test_function_preserving_r2deeperr(model, thresh, data_channels=1, layer1=None, layer2=None, verbose=False,
                                       resnet=False, spatial_dim=32):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])
    l1_norm_before = parameter_magnitude(model)

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-0.5, high=0.5, size=(1,data_channels,spatial_dim,spatial_dim)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # deepen and check that the outputs are (almost) identical
    if not resnet:
        model = make_deeper_network_(model, layer1)
    else:
        model.deepen([2,2,0,0])
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.mean(t.abs(rand_out - rand_outs[i]))
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    l1_norm_after = parameter_magnitude(model)
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))
        print("Params ratio before and after the transform is: {param}".format(param=params_after/params_before))
        print("Param L1 before the transform is: {param}".format(param=l1_norm_before))
        print("Param L1 after the transform is: {param}".format(param=l1_norm_after))
        print("Param L1 ratio before and after the transform is: {param}".format(param=l1_norm_after/l1_norm_before))

    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    if not resnet:
        model = make_deeper_network_(model, layer2)
    else:
        model.deepen([0,1,2,0])

    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.max(t.abs(rand_out - rand_outs[i]))
        if verbose:
            print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")



def test_function_preserving_r2widerr(model, thresh, function_preserving=True, data_channels=1, verbose=False, deep=False, spatial_dim=32, net_morph=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])
    l1_norm_before = parameter_magnitude(model)

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-0.5, high=0.5, size=(1,data_channels,spatial_dim,spatial_dim)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    if not deep:
        model = widen_network_(model, new_channels=4, new_hidden_nodes=2, init_type='He',
                               function_preserving=function_preserving, multiplicative_widen=True, net_morph=net_morph)
    else:
        model = widen_network_(model, new_channels=2, new_hidden_nodes=2, init_type='match_std_exact',
                               function_preserving=function_preserving, multiplicative_widen=True, net_morph=net_morph)
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.max(t.abs(rand_out - rand_outs[i]))
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    l1_norm_after = parameter_magnitude(model)
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))
        print("Params ratio before and after the transform is: {param}".format(param=params_after/params_before))
        print("Param L1 before the transform is: {param}".format(param=l1_norm_before))
        print("Param L1 after the transform is: {param}".format(param=l1_norm_after))
        print("Param L1 ratio before and after the transform is: {param}".format(param=l1_norm_after/l1_norm_before))

    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    model = widen_network_(model, new_channels=16, new_hidden_nodes=16, init_type='match_std_exact',
                           function_preserving=function_preserving, multiplicative_widen=False, net_morph=net_morph)


    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.max(t.abs(rand_out - rand_outs[i]))
        if verbose:
            print("Avg output difference before and after ANOTHER transform is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")
        
        
        
def test_function_preserving_deepen_then_widen(model, thresh, function_preserving=True, data_channels=1, layer=None,
                                               verbose=False, spatial_dim=32, resnet=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-0.5, high=0.5, size=(1,data_channels,spatial_dim,spatial_dim)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    if not resnet:
        model = make_deeper_network_(model, layer)
        model = widen_network_(model, new_channels=4, new_hidden_nodes=2, init_type='He',
                               function_preserving=function_preserving, multiplicative_widen=True)
    else:
        model.deepen([2,2,0,0])
        model = widen_network_(model, new_channels=2, new_hidden_nodes=2, init_type='match_std_exact',
                               function_preserving=function_preserving, multiplicative_widen=True)
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.max(t.abs(rand_out - rand_outs[i]))
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
        if t.abs(err) > thresh:
            raise Exception("Unit test failed.")
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    if verbose:
        print("Params before the transform is: {param}".format(param=params_before))
        print("Params after the transform is: {param}".format(param=params_after))
        
        
def test_function_preserving_widen_then_deepen(model, thresh, function_preserving=True, data_channels=1, layer=None,
                                               verbose=False, spatial_dim=32, resnet=False):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-0.5, high=0.5, size=(1,data_channels,spatial_dim,spatial_dim)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (multiplicative_widen) and check that the outputs are (almost) identical
    if not resnet:
        model = widen_network_(model, new_channels=4, new_hidden_nodes=2, init_type='He',
                               function_preserving=function_preserving, multiplicative_widen=True)
        model = make_deeper_network_(model, layer)
    else:
        model = widen_network_(model, new_channels=2, new_hidden_nodes=2, init_type='match_std_exact',
                               function_preserving=function_preserving, multiplicative_widen=True)
        model.deepen([2,2,0,0])
    
    for i in range(10):
        rand_out = model(rand_ins[i])
        err = t.max(t.abs(rand_out - rand_outs[i]))
        if verbose:
            print("Average output difference before and after transform is: {val}".format(val=err))
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c11 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.c21 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.c22 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.linear1 = nn.Linear((20 + 20) * 16 * 16, 10)
        self.linear2 = nn.Linear(10, 10)

    def conv_forward(self, x):
        x = self.pool(x)
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
        return (1, 16, 16)

    def conv_hvg(self, cur_hvg):
        root_node = cur_hvg.get_output_nodes()[0]
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c11.weight.size(0) + self.c12.weight.size(0), 16, 16),
                                   input_modules=[self.c11, self.c12],
                                   input_hvns=[root_node, root_node])
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c21.weight.size(0) + self.c22.weight.size(0), 16, 16),
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





class _Baby_Inception(nn.Module):
    """
    A small siamese network, with 2 pathways, just to stress test 
    """
    def __init__(self, test_concat=False, add_batch_norm=True):
        super(_Baby_Inception, self).__init__()
        self.c11 = nn.Conv2d(1, 10, kernel_size=1, padding=0)
        self.bn11 = nn.BatchNorm2d(num_features=10) if add_batch_norm else None
        self.c12 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features=6) if add_batch_norm else None
        self.c13 = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.bn13 = nn.BatchNorm2d(num_features=4) if add_batch_norm else None
        self.c21 = nn.Conv2d(20, 10, kernel_size=1, padding=0)
        self.c22 = nn.Conv2d(20, 10, kernel_size=3, padding=1)
        self.c23 = nn.Conv2d(20, 10, kernel_size=5, padding=2)
        self.c24 = nn.Conv2d(20, 10, kernel_size=7, padding=3)
        self.pool31 = nn.MaxPool2d(kernel_size=2)
        self.c32 = nn.Conv2d(40, 20, kernel_size=2, padding=0, stride=2)
        self.pool41 = nn.MaxPool2d(kernel_size=2)
        self.c51 = nn.Conv2d(60, 60, kernel_size=1, padding=0)
        self.linear1 = nn.Linear((10+10+10+10+20)*8*8, 10)
        self.linear2 = nn.Linear(10, 10)

        self.test_concat = test_concat
        
    def conv_forward(self, x):
        x1 = self.c11(x)
        if self.bn11:
            x1 = self.bn11(x1)
        x2 = self.c12(x)
        if self.bn12:
            x2 = self.bn12(x2)
        x3 = self.c13(x)
        if self.bn13:
            x3 = self.bn13(x3)
        x = t.cat((x1,x2,x3), 1)
        x1 = self.c21(x)
        x2 = self.c22(x)
        x3 = self.c23(x)
        x4 = self.c24(x)
        x = t.cat((x1,x2,x3,x4), 1)
        x1 = self.pool31(x)
        x2 = self.c32(x)
        x = t.cat((x1,x2), 1)
        x = self.pool41(x)
        x = self.c51(x)
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
        return (1,32,32)
    
    def conv_hvg(self, cur_hvg):
        root_node = cur_hvg.get_output_nodes()[0]
        if not self.test_concat:
            cur_node = cur_hvg.add_hvn(hv_shape=(self.c11.weight.size(0)+self.c12.weight.size(0)+self.c13.weight.size(0), 32, 32),
                                       input_modules=[self.c11, self.c12, self.c13],
                                       input_hvns=[root_node, root_node, root_node],
                                       batch_norm=[self.bn11, self.bn12, self.bn13])
        else:
            node1 = cur_hvg.add_hvn(hv_shape=(self.c11.weight.size(0), 32, 32),
                                       input_modules=[self.c11],
                                       input_hvns=[root_node],
                                       batch_norm=self.bn11)
            node2 = cur_hvg.add_hvn(hv_shape=(self.c12.weight.size(0), 32, 32),
                                       input_modules=[self.c12],
                                       input_hvns=[root_node],
                                       batch_norm=self.bn12)
            node3 = cur_hvg.add_hvn(hv_shape=(self.c13.weight.size(0), 32, 32),
                                       input_modules=[self.c13],
                                       input_hvns=[root_node],
                                       batch_norm=self.bn13)
            cur_node = cur_hvg.concat([node1, node2, node3])

        cur_node = cur_hvg.add_hvn(hv_shape=(self.c21.weight.size(0)+self.c22.weight.size(0)+self.c23.weight.size(0)+self.c24.weight.size(0), 32, 32),
                                   input_modules=[self.c21, self.c22, self.c23, self.c24],
                                   input_hvns=[cur_node, cur_node, cur_node, cur_node])
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c21.weight.size(0)+self.c22.weight.size(0)+self.c23.weight.size(0)+self.c24.weight.size(0)+self.c32.weight.size(0), 16, 16),
                                   input_modules=[self.pool31, self.c32],
                                   input_hvns=[cur_node, cur_node])
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c21.weight.size(0)+self.c22.weight.size(0)+self.c23.weight.size(0)+self.c24.weight.size(0)+self.c32.weight.size(0), 8, 8),
                                   input_modules=[self.pool41])
        cur_node = cur_hvg.add_hvn(hv_shape=(self.c51.weight.size(0), 8, 8),
                                   input_modules=[self.c51])
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
    if verbose:
        print("Testing Res_Block identity initialize:")
    test_res_block_identity_initialize()

    if verbose:
        print("\n"*4)
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
    rblock = Res_Block(input_channels=32, intermediate_channels=[32,32,32], output_channels=32,
                       identity_initialize=True, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=32, intermediate_channels=[32,32,32], output_channels=32,
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Mnist_Resnet(), 1e-5, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for Mnist Resnet:")
    rblock = Res_Block(input_channels=32, intermediate_channels=[32,32,32], output_channels=32,
                       identity_initialize=False, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=32, intermediate_channels=[32,32,32], output_channels=32,
                       identity_initialize=False, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Mnist_Resnet(), 1e5, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[32,32,32], output_channels=64,
                       identity_initialize=True, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=64, intermediate_channels=[32,32,32], output_channels=64,
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Cifar_Resnet(), 1e-5, data_channels=3, layer1=rblock, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[32,32,32], output_channels=64,
                       identity_initialize=False, input_spatial_shape=(4,4))
    rblock2 = Res_Block(input_channels=64, intermediate_channels=[32,32,32], output_channels=64,
                       identity_initialize=False, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Cifar_Resnet(), 1e5, data_channels=3, layer1=rblock, layer2=rblock2, verbose=verbose)



    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2DeeperR + R2WiderR for Mnist Resnet:")
    # rblock = Res_Block(input_channels=32, intermediate_channels=[32,32,32], output_channels=32,
    #                    identity_initialize=True, input_spatial_shape=(4,4))
    # test_function_preserving_deepen_then_widen(Mnist_Resnet(), 1e-5, layer=rblock, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2DeeperR + R2WiderR for Cifar Resnet:")
    # rblock = Res_Block(input_channels=64, intermediate_channels=[32,32,32], output_channels=64,
    #                    identity_initialize=True, input_spatial_shape=(4,4))
    # test_function_preserving_deepen_then_widen(Cifar_Resnet(), 1e-5, data_channels=3, layer=rblock, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2WiderR + R2DeeperR for Mnist Resnet:")
    # # insert wider resblock, because widen before deepen this time
    # rblock = Res_Block(input_channels=64, intermediate_channels=[32,32,32], output_channels=64,
    #                    identity_initialize=True, input_spatial_shape=(4,4))
    # test_function_preserving_widen_then_deepen(Mnist_Resnet(), 1e-5, layer=rblock, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2WiderR + R2DeeperR for Cifar Resnet:")
    # # insert wider resblock, because widen before deepen this time
    # rblock = Res_Block(input_channels=128, intermediate_channels=[32,32,32], output_channels=128,
    #                    identity_initialize=True, input_spatial_shape=(4,4))
    # test_function_preserving_widen_then_deepen(Cifar_Resnet(), 1e-5, data_channels=3, layer=rblock, verbose=verbose)



    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for Siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese(), 1e-5, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding widening for Siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese(), 1e5, False, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for Siamese network:")
    rblock1 = Res_Block(input_channels=40, intermediate_channels=[20,20,20], output_channels=40,
                       identity_initialize=True, input_spatial_shape=(16,16))
    rblock2 = Res_Block(input_channels=40, intermediate_channels=[20,20,20], output_channels=40,
                       identity_initialize=True, input_spatial_shape=(16,16))
    test_function_preserving_r2deeperr(_Baby_Siamese(), 1e-5, layer1=rblock1, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for Siamese network:")
    rblock1 = Res_Block(input_channels=40, intermediate_channels=[32,32,32], output_channels=40,
                       identity_initialize=False, input_spatial_shape=(16,16))
    rblock2 = Res_Block(input_channels=40, intermediate_channels=[32,32,32], output_channels=40,
                       identity_initialize=False, input_spatial_shape=(16,16))
    test_function_preserving_r2deeperr(_Baby_Siamese(), 1e5, layer1=rblock1, layer2=rblock2, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2DeeperR + R2WiderR for Siamese Network:")
    # rblock = Res_Block(input_channels=40, intermediate_channels=[20,20,20], output_channels=40,
    #                    identity_initialize=True, input_spatial_shape=(16,16))
    # test_function_preserving_deepen_then_widen(_Baby_Siamese(), 1e-5, layer=rblock, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2WiderR + R2DeeperR for Siamese Network:")
    # rblock = Res_Block(input_channels=80, intermediate_channels=[20,20,20], output_channels=80,
    #                    identity_initialize=True, input_spatial_shape=(32,32))
    # test_function_preserving_widen_then_deepen(_Baby_Siamese(), 1e-5, layer=rblock, verbose=verbose)



    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for Baby Inception network:")
    test_function_preserving_r2widerr(_Baby_Inception(), 1e-5, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR hvg.concat works:")
    test_function_preserving_r2widerr(_Baby_Inception(test_concat=True), 1e-4, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding widening for Baby Inception network:")
    test_function_preserving_r2widerr(_Baby_Inception(), 1e5, False, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for Baby Inception network:")
    rblock1 = Res_Block(input_channels=60, intermediate_channels=[20,20,20], output_channels=60,
                       identity_initialize=True, input_spatial_shape=(8,8))
    rblock2 = Res_Block(input_channels=60, intermediate_channels=[20,20,20], output_channels=60,
                       identity_initialize=True, input_spatial_shape=(8,8))
    test_function_preserving_r2deeperr(_Baby_Inception(), 1e-5, layer1=rblock1, layer2=rblock2, verbose=verbose)

    if verbose:
        print("\n"*4)
        print("Testing random padding deepening for Baby Inception network:")
    rblock1 = Res_Block(input_channels=60, intermediate_channels=[32,32,32], output_channels=60,
                       identity_initialize=False, input_spatial_shape=(8,8))
    rblock2 = Res_Block(input_channels=60, intermediate_channels=[32,32,32], output_channels=60,
                       identity_initialize=False, input_spatial_shape=(8,8))
    test_function_preserving_r2deeperr(_Baby_Inception(), 1e5, layer1=rblock1, layer2=rblock2, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2DeeperR + R2WiderR for Baby Inception Network:")
    # rblock = Res_Block(input_channels=60, intermediate_channels=[20,20,20], output_channels=60,
    #                    identity_initialize=True, input_spatial_shape=(8,8))
    # test_function_preserving_deepen_then_widen(_Baby_Inception(), 1e-5, layer=rblock, verbose=verbose)

    # if verbose:
    #     print("\n"*4)
    #     print("Testing R2WiderR + R2DeeperR for Baby Inception Network:")
    # rblock = Res_Block(input_channels=120, intermediate_channels=[20,20,20], output_channels=120,
    #                    identity_initialize=True, input_spatial_shape=(8,8))
    # test_function_preserving_widen_then_deepen(_Baby_Inception(), 1e-5, layer=rblock, verbose=verbose)



    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for Inception network:")
    test_function_preserving_r2widerr(InceptionV4().eval(), 1e-4, verbose=verbose, data_channels=3, deep=True, spatial_dim=299)

    if verbose:
        print("\n"*4)
        print("Testing random padding for Inception network:")
    test_function_preserving_r2widerr(InceptionV4(), 1e20, False, verbose=verbose, data_channels=3, deep=True, spatial_dim=299)



    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for ResNet50 network:")
    test_function_preserving_r2widerr(resnet50(), 1e-4, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing random padding for ResNet50 network:")
    test_function_preserving_r2widerr(resnet50(), 1e20, False, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for ResNet18 network:")
    test_function_preserving_r2widerr(resnet18(), 1e-4, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing random padding for ResNet18 network:")
    test_function_preserving_r2widerr(resnet18(), 1e20, False, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for ResNet18 network:")
    test_function_preserving_r2deeperr(resnet18(), 1e-4, verbose=verbose, data_channels=3, resnet=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing random padding on deepening for ResNet18 network:")
    test_function_preserving_r2deeperr(resnet18(function_preserving=False), 1e20, verbose=verbose, data_channels=3, resnet=True, spatial_dim=224)



    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for ResNet50 network:")
    test_function_preserving_r2widerr(resnet50(thin=True, thinning_ratio=16*1.414), 1e-4, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing random padding for ResNet50 network:")
    test_function_preserving_r2widerr(resnet50(thin=True, thinning_ratio=16*1.414), 1e20, False, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing R2WiderR for ResNet18 network:")
    test_function_preserving_r2widerr(resnet18(thin=True, thinning_ratio=16), 1e-5, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing random padding for ResNet18 network:")
    test_function_preserving_r2widerr(resnet18(thin=True, thinning_ratio=16*1.414), 1e20, False, verbose=verbose, data_channels=3, deep=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing R2DeeperR for ResNet18 network:")
    test_function_preserving_r2deeperr(resnet18(thin=True, thinning_ratio=16*1.414), 1e-5, verbose=verbose, data_channels=3, resnet=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing random padding on deepening for ResNet18 network:")
    test_function_preserving_r2deeperr(resnet18(function_preserving=False, thin=True, thinning_ratio=16*1.414), 1e20, verbose=verbose, data_channels=3, resnet=True, spatial_dim=224)
    
    if verbose:
        print("\n"*4)
        print("Testing deeoeb then widen on deepening for ResNet50 network:")
        test_function_preserving_widen_then_deepen(resnet50(), 1e-3, verbose=verbose, data_channels=3, resnet=True, spatial_dim=224)

    if verbose:
        print("\n"*4)
        print("Testing widen then deepen on deepening for ResNet50 network:")
    test_function_preserving_deepen_then_widen(resnet50(), 1e-3, verbose=verbose, data_channels=3, resnet=True, spatial_dim=224)
    
    

















    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet().eval(), 1e-5, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet().eval(), 1e5, False, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet().eval(), 1e-5, data_channels=3, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet().eval(), 1e5, False, data_channels=3, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for Siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese().eval(), 1e-5, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding widening for Siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese().eval(), 1e5, False, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for Baby Inception network:")
    test_function_preserving_r2widerr(_Baby_Inception().eval(), 1e-5, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding widening for Baby Inception network:")
    test_function_preserving_r2widerr(_Baby_Inception().eval(), 1e5, False, verbose=verbose, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for Inception network:")
    test_function_preserving_r2widerr(InceptionV4().eval(), 1e-4, verbose=verbose, data_channels=3, deep=True,
                                      spatial_dim=299, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding for Inception network:")
    test_function_preserving_r2widerr(InceptionV4().eval(), 1e20, False, verbose=verbose, data_channels=3, deep=True,
                                      spatial_dim=299, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for ResNet50 network:")
    test_function_preserving_r2widerr(resnet50().eval(), 1e-4, verbose=verbose, data_channels=3, deep=True, spatial_dim=224, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding for ResNet50 network:")
    test_function_preserving_r2widerr(resnet50().eval(), 1e20, False, verbose=verbose, data_channels=3, deep=True,
                                      spatial_dim=224, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing NetMorph Widening for ResNet18 network:")
    test_function_preserving_r2widerr(resnet18().eval(), 1e-4, verbose=verbose, data_channels=3, deep=True, spatial_dim=224, net_morph=True)

    if verbose:
        print("\n" * 4)
        print("Testing random padding for ResNet18 network:")
    test_function_preserving_r2widerr(resnet18().eval(), 1e20, False, verbose=verbose, data_channels=3, deep=True,
                                      spatial_dim=224, net_morph=True)