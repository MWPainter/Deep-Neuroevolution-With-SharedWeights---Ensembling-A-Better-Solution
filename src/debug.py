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

    student2 = resnet35(thin=True, thinning_ratio=1.5)
    student2.function_preserving = False
    student2.init_scheme = 'He'
    student2.deepen([1,1,2,1])
    student2.widen(1.5)

    student_m1 = resnet35(thin=True, thinning_ratio=1.5)
    student_m1.function_preserving = False
    student_m1.init_scheme = 'match_std_exact'
    student_m1.widen(1.5)
    student_m1.deepen([1,1,2,1])

    student_m2 = resnet35(thin=True, thinning_ratio=1.5)
    student_m2.function_preserving = False
    student_m2.init_scheme = 'match_std_exact'
    student_m2.deepen([1,1,2,1])
    student_m2.widen(1.5)

    resnet = resnet50()

    print("Student network num params:")
    print(count_parameters(student))
    print("Student network weight magnitude:")
    print(parameter_magnitude(student))

    print()
    print("Student2 network num params:")
    print(count_parameters(student2))
    print("Student2 network weight magnitude:")
    print(parameter_magnitude(student2))

    print()
    print("Student_m1 network num params:")
    print(count_parameters(student_m1))
    print("Student_m1 network weight magnitude:")
    print(parameter_magnitude(student_m1))

    print()
    print("Student_m2 network num params:")
    print(count_parameters(student_m2))
    print("Student_m2 network weight magnitude:")
    print(parameter_magnitude(student_m2))

    print()
    print("Resnet num params:")
    print(count_parameters(resnet))
    print("Resnet weight magnitude:")
    print(parameter_magnitude(resnet))

