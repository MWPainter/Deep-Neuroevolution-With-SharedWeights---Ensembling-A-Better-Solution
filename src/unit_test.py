import numpy as np
import torch as t
import torch.nn as nn

from r2r import Mnist_Resnet, Cifar_Resnet, widen_network_


    
    
    
def test_function_preserving_r2deeperr(model, thresh, function_preserving=True, data_channels=1, layer=None):
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
    network = make_deeper_network(network, layer)
    
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Average output difference before and after transform is: {val}".format(val=t.mean(rand_out - rand_outs[i])))
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    print("Params before the transform is: {param}".format(param=params_before))
    print("Params after the transform is: {param}".format(param=params_after)) 

    # widen (scaled) and check that the outputs are (almost) identical
    network = make_deeper_network(network, layer)
    
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Avg output difference before and after ANOTHER transform is: {val}".format(val=t.mean(rand_out - rand_outs[i])))



def test_function_preserving_r2widerr(model, thresh, function_preserving=True, data_channels=1):
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
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Average output difference before and after transform is: {val}".format(val=t.mean(rand_out - rand_outs[i])))
    
    # Count params after widening
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    print("Params before the transform is: {param}".format(param=params_before))
    print("Params after the transform is: {param}".format(param=params_after)) 

    # widen (scaled) and check that the outputs are (almost) identical
    model = widen_network_(model, new_channels=4, new_hidden_nodes=2, init_type='He', 
                           function_preserving=function_preserving, scaled=True)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Avg output difference before and after ANOTHER transform is: {val}".format(val=t.mean(rand_out - rand_outs[i])))
    
    
    
    
class _Baby_Siamese(nn.Module):
    """
    A small siamese network, with 2 pathways, just to stress test 
    """
    def __init__(self):
        super(_Baby_Siamese, self).__init__()
        self.c11 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.c21 = nn.Conv2d(20, , kernel_size=3, padding=1)
        self.c22 = nn.Conv2d(20, intermediate_channels[0], kernel_size=3, padding=1)
        self.linear = nn.Linear((20+20)*32*32, 2)
        
    def conv_forward(self, x):
        x1 = self.c11(x)
        x2 = self.c12(x)
        x = 
        x1 = self.c21(x)
        x2 = self.c22(x)
        x = 
        return x
        
    def fc_forward(self, x):
        x = flatten(x)
        x = self.linear(x)
        return x
        
    def out_forward(self, x):
        return x
    
    def forward(self, x):
        x = self.conv_forward(x)
        x = self.fc_forward(x)
        return self.out_forward(x)
    
    def conv_hvg(self):
        # TODO
        pass
    
    def fc_hvg(self):
        # TODO
        pass
    
    def hvg(self):
        # TODO
        pass
    
    
    
    
    

if __name__ == "__main__":
    print("Testing R2WiderR for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet(), 0.0001)
    
    print("\n"*4)
    print("Testing random padding for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet(), 0.0001, False)
    
    print("\n"*4)
    print("Testing R2WiderR for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet(), 0.0001, data_channels=3)
    
    print("\n"*4)
    print("Testing random padding for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet(), 0.0001, False, data_channels=3)
    
    print("Testing R2DeeperR for Mnist Resnet:")
    rblock = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Mnist_Resnet(), 0.0001, layer=rblock)
    
    print("\n"*4)
    print("Testing random padding deepening for Mnist Resnet:")
    rblock = Res_Block(input_channels=32, intermediate_channels=[2,2,2], output_channels=32, 
                       identity_initialize=False, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Mnist_Resnet(), 0.0001, False, layer=rblock)
    
    print("\n"*4)
    print("Testing R2DeeperR for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=True, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Cifar_Resnet(), 0.0001, data_channels=3, layer=rblock)
    
    print("\n"*4)
    print("Testing random padding deepening for Cifar Resnet:")
    rblock = Res_Block(input_channels=64, intermediate_channels=[2,2,2], output_channels=64, 
                       identity_initialize=False, input_spatial_shape=(4,4))
    test_function_preserving_r2deeperr(Cifar_Resnet(), 0.0001, False, data_channels=3, layer=rblock)
    
    print("\n"*4)
    print("Testing R2WiderR for siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese(), 0.0001)
    
    print("\n"*4)
    print("Testing random padding for siamese network:")
    test_function_preserving_r2widerr(_Baby_Siamese(), 0.0001, False)
    
    print("\n"*4)
    print("Testing R2DeeperR for siamese network:")
    rblock = Res_Block(input_channels=40, intermediate_channels=[2,2,2], output_channels=40, 
                       identity_initialize=True, input_spatial_shape=(32,32))
    test_function_preserving_r2deeperr(_Baby_Siamese(), 0.0001, layer=rblock)
    
    print("\n"*4)
    print("Testing random padding deepening for Cifar Resnet:")
    rblock = Res_Block(input_channels=40, intermediate_channels=[2,2,2], output_channels=40, 
                       identity_initialize=False, input_spatial_shape=(32,32))
    test_function_preserving_r2deeperr(_Baby_Siamese(), 0.0001, False, layer=rblock)
    
    