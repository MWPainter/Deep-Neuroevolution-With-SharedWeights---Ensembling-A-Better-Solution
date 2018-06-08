from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import torch as t

from dataset import *

ROOT = os.path.abspath(os.path.join('.', 'dataset', 'data', 'cifar10'))

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10():
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10()

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }









class CifarDataset(Dataset):
    """
    Above are helper functions, taken from the cs231n starter code (publicly available :)) 
    We wrangle them into a sublass of the Dataset object we've defined.
    """
    def __init__(self, batch_size=32, subtract_mean=True, validation_set_size=1000):
        Dataset.__init__(self)
        
        self.name = "cifar"
        
        self.data_dims = [32, 32, 3]
        self.range = [0.0, 1.0]
        self.batch_size = batch_size
        self.subtract_mean = subtract_mean
        
        data = get_data(num_training=50000-validation_set_size,
                        num_validation=validation_set_size,
                        subtract_mean=subtract_mean)
        self.xtrain = data['X_train']
        self.ytrain = data['y_train']
        self.xval = data['X_val']
        self.yval = data['y_val']
        self.xtest = data['X_test']
        self.ytest = data['y_test']
        
        self.validation_set_size = validation_set_size
        self.test_set_size = self.xtest.shape[0]
        
    def _indices(self, batch_size, max_index):
        return np.random.choice(max_index, batch_size, replace=True)
    
    def _one_hot(self, classes, num_classes=10):
        return np.eye(num_classes)[classes]

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = self._indices(batch_size, self.xtrain.shape[0])
        imgs = t.tensor(self.xtrain[indices], dtype=t.float)
        labels = t.tensor(self._one_hot(self.ytrain[indices]), dtype=t.float)
        return imgs, labels
    
    def next_val_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = self._indices(batch_size, self.xval.shape[0])
        imgs = t.tensor(self.xval[indices], dtype=t.float)
        labels = t.tensor(self._one_hot(self.yval[indices]), dtype=t.float)
        return imgs, labels
    
    def get_val_set_size(self):
        return self.validation_set_size
    
    def val_set(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        i = 0
        while i < self.validation_set_size:
            xs = self.xval[i:i+batch_size]
            ys = self.yval[i:i+batch_size]
            yield t.tensor(xs, dtype=t.float), t.tensor(self._one_hot(ys), dtype=t.float)
            i += batch_size
    
    def get_test_set_size(self):
        return self.test_set_size
        
    def test_set(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        i = 0
        while i < self.test_set_size:
            xs = self.xtest[i:i+batch_size]
            ys = self.ytest[i:i+batch_size]
            yield t.tensor(xs, dtype=t.float), t.tensor(self._one_hot(ys), dtype=t.float)
            i += batch_size

    def next_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = self._indices(batch_size, self.xtest.shape[0])
        imgs = t.tensor(self.xtest[indices], dtype=t.float)
        labels = t.tensor(self._one_hot(self.ytest[indices]), dtype=t.float)
        return imgs, labels

    def display(self, image):
        return np.clip(image, 0.0, 1.0)

    def reset(self):
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0

if __name__ == '__main__':
    dataset = CifarDataset()
    images = dataset.next_batch()
    for i in range(100):
        plt.imshow(dataset.display(images[i]))
        plt.show()