import numpy as np
import math
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from glob import glob
from scipy import misc

try:        # Works for python 3
    from dataset.dataset import *
    from dataset.dataset_mnist import MnistDataset
except:     # Works for python 2
    from dataset import *
    from dataset_mnist import MnistDataset
