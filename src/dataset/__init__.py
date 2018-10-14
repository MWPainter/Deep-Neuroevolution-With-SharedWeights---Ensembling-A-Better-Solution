import numpy as np
import math
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from glob import glob
from scipy import misc

try:        # Works for python 3
    from dataset.dataset import Dataset
    from dataset.dataset import DatasetCudaWrapper
    from dataset.dataset_mnist import MnistDataset
    from dataset.dataset_cifar import CifarDataset
except:     # Works for python 2
    from dataset import Dataset
    from dataset import DatasetCudaWrapper
    from dataset_mnist import MnistDataset
    from dataset_cifar import CifarDataset