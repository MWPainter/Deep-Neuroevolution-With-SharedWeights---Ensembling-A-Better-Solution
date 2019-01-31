import numpy as np
import torch as t
import torch.nn as nn

from r2r import *
from utils import *
    
    
    
    
    

if __name__ == "__main__":
    student = resnet35(thin=True, thinning_ratio=1.5)
    student.function_preserving = False
    student.init_scheme = 'He'
    student.widen(1.5)
    student.deepen([1,1,2,1])

    resnet = resnet50()

    print("Student network num params:")
    print(count_parameters(student))
    print("Student network weight magnitude:")
    print(parameter_magnitude(student))

    print()
    print("Resnet num params:")
    print(count_parameters(resnet))
    print("Resnet weight magnitude:")
    print(parameter_magnitude(resnet))

