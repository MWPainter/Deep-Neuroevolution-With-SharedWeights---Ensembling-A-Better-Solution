from dataset import *
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import torch as t


class MnistDataset(Dataset):
    def __init__(self, batch_size=32, subtract_mean=True, binary=False):
        Dataset.__init__(self)
        data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'mnist')
        self.mnist = input_data.read_data_sets(data_file, one_hot=True)
        self.name = "mnist"
        
        self.data_dims = [1, 32, 32] # actually 28x28, but added padding so it's a power of 2 and nice for the LVAE
        self.train_size = 50000
        self.val_size = 10000
        self.width = 32
        self.height = 32
        self.binary = binary
        
        self.range = [0.0, 1.0]
        self.batch_size = batch_size
        self.subtract_mean = subtract_mean
        
        if self.subtract_mean:
            xs, _ = self.mnist.train.next_batch(self.train_size)
            self.mean = np.mean(xs)

    def next_batch(self, batch_size=None):
        if batch_size is None: 
            batch_size = self.batch_size
        
        xs, labels = self.mnist.train.next_batch(batch_size)
        images = np.reshape(xs, (-1, 1, 28, 28))
        images = np.reshape(images, (-1, 1, 28, 28))
        images = np.pad(images, ((0,0),(0,0),(2,2),(2,2)), 'edge') # pads images to be 32x32 vs 28x28
        
        if self.subtract_mean:
            images -= self.mean
        
        if self.binary:
            return t.tensor(np.rint(images)), t.tensor(labels, dtype=t.float)
        else:
            return t.tensor(images), t.tensor(labels, dtype=t.float)

    def next_val_batch(self, batch_size=None):
        if batch_size is None: 
            batch_size = self.batch_size
            
        xs, labels = self.mnist.test.next_batch(batch_size)
        images = np.reshape(xs, (-1, 1, 28, 28))
        images = np.pad(images, ((0,0),(0,0),(2,2),(2,2)), 'edge') # pads images to be 32x32 vs 28x28
        
        if self.subtract_mean:
            images -= self.mean
        
        if self.binary:
            return t.tensor(np.rint(images)), t.tensor(labels, dtype=t.float)
        else:
            return t.tensor(images), t.tensor(labels, dtype=t.float)
        
    
    def next_test_batch(self, batch_size):
        """
        DON'T USE MNIST FOR TESTING... IT'S TOO SMALL OF A DATASET...
        """
        self.handle_unsupported_op()
        return None

    def display(self, image):
        return np.clip(image, a_min=0.0, a_max=1.0)

    def reset(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


if __name__ == '__main__':
    binary_data = MnistDataset(binary=True)
    float_data = MnistDataset(binary=False)
    while True:
        binary_sample = binary_data.next_batch(100)
        float_sample = float_data.next_batch(100)
        for index in range(9):
            plt.subplot(3, 6, 2 * index + 1)
            plt.imshow(float_sample[index, :, :, 0].astype(np.float), cmap=plt.get_cmap('Greys'))
            plt.subplot(3, 6, 2 * index + 2)
            plt.imshow(binary_sample[index, :, :, 0].astype(np.float), cmap=plt.get_cmap('Greys'))
        plt.show()
