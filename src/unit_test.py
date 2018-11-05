import numpy as np
import torch as t

from r2r import Mnist_Resnet, Cifar_Resnet, widen_network_






def test_function_preserving_r2widerr(model, thresh, function_preserving=True):
    # Count params before widening
    params_before = sum([np.prod(p.size()) for p in model.parameters()])

    # store 10 random inputs, and their outputs
    rand_ins = []
    rand_outs = []
    for _ in range(10):
        rand_in = t.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(1,1,32,32)))
        rand_out = model(rand_in)
        rand_ins.append(rand_in)
        rand_outs.append(rand_out)

    # widen (scaled) and check that the outputs are (almost) identical
    widen_network_(model, new_channels=2, new_hidden_nodes=3, init_type='He', function_preserving=function_preserving, 
                   scaled=True)
    _widen_network_convs_scaled_(model, 2)
    params_after = sum([np.prod(p.size()) for p in model.parameters()])
    for i in range(10):
        rand_out = model(rand_ins[i])
        print("Average output difference before and after widen is: %f" % t.mean(rand_out - rand_outs[i]))
    
    
    
    

if __name__ == "__main__":
    print("Testing R2WiderR for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet(), 0.0001)
    
    print("\n"*4)
    print("Testing random padding for Mnist Resnet:")
    test_function_preserving_r2widerr(Mnist_Resnet(), 0.0001, False)
    
    print("\n"*4)
    print("Testing R2WiderR for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet(), 0.0001)
    
    print("\n"*4)
    print("Testing random padding for Cifar Resnet:")
    test_function_preserving_r2widerr(Cifar_Resnet(), 0.0001, False)