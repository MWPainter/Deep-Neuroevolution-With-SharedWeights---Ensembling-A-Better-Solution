import numpy as np
import torch as t

from r2r import resnet50


if __name__ == "__main__":
    model = resnet50().cuda()
    batch_size = 1
    while True:
        rand_in = t.Tensor(np.random.uniform(low=-0.5, high=0.5, size=(batch_size,3,224,224))).cuda()
        rand_out = model(rand_in)
        print("Can run batch size of {bs}.".format(bs=batch_size))
        batch_size = batch_size ** 2
